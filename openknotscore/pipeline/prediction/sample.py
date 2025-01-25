from typing import Callable, Any
import traceback
from dataclasses import dataclass
import itertools
import math
import random
import time
import multiprocessing
import threading
import resource
import psutil
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from .predictors import Predictor
from pynvml_utils import nvidia_smi

@dataclass
class ResourceSample:
    time: float
    maxrss: int
    maxgpu: int
    result: Any

@dataclass
class AggregateResourceSample:
    seqlen: int
    min_runtime: float
    average_runtime: float
    max_runtime: float
    max_memory: float
    max_gpu: float

def sample_memory(stop: threading.Event, out: multiprocessing.Queue):
    max_mem = 0
    current_process = psutil.Process()
    while not stop.wait(0.05):     
        mem = current_process.memory_info().rss
        for child in current_process.children(recursive=True):
            mem += child.memory_info().rss
        max_mem = max(mem, max_mem)
    out.put(max_mem)

def sample_gpu(stop: threading.Event, out: multiprocessing.Queue):
    max_mem = 0
    nvsmi = nvidia_smi.getInstance()
    while not stop.wait(0.05):
        mem = nvsmi.DeviceQuery('memory.used')['gpu'][0]['fb_memory_usage']['used'] * 1024 * 1024
        max_mem = max(mem, max_mem)
    out.put(max_mem)

def run_sample(f, out: multiprocessing.Queue, gpu: bool):
    done = threading.Event()
    mem_sampler_out = multiprocessing.Queue()
    mem_sampler = threading.Thread(target=sample_memory, args=(done,mem_sampler_out))
    mem_sampler.start()
    if gpu:
        gpu_sampler_out = multiprocessing.Queue()
        gpu_sampler = threading.Thread(target=sample_gpu, args=(done,gpu_sampler_out))
        gpu_sampler.start()

    start = time.perf_counter()
    try:
        res = f()
    except Exception as e:
        traceback.print_exc()
        res = None
    end = time.perf_counter()
    
    done.set()
    mem_sampler.join()
    max_mem = mem_sampler_out.get()
    if gpu:
        gpu_sampler.join()
        max_gpu = gpu_sampler_out.get()
    else:
        max_gpu = 0

    out.put(
        ResourceSample(
            end - start,
            max_mem,
            max_gpu,
            res
        )
    )

def sample_in_process(f, timeout, gpu) -> ResourceSample:
    sampler_out = multiprocessing.Queue()
    p = multiprocessing.Process(target=run_sample, args=(f, sampler_out, gpu), daemon=True)
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join(5)
        if p.is_alive():
            p.kill()
        raise TimeoutError()
    
    return sampler_out.get()

# We want a runtime of at least MIN_TIMEOUT seconds to be relatively confident that our benchmarks
# are showing useful signal rather than noise, but we bail and try a smaller size if its been
# running longer than MAX_TIMEOUT. This way we can aggressively increase our sequence length where
# possible for very fast algorithms, but also not sit around forever if increasing that large of
# a step for a slow algorithm causes it to run for a very long time
MIN_TIMEOUT = 30
MAX_TIMEOUT = 90
# If we can run a strand this long within our timeout, just give up. We may be doing something
# like using a constant-time function as a mock
MAX_LEN = 25000

MIN_SAMPLES = 10

class FullSampleFailure(Exception):
    pass

def sample_resources_at_length(predictor: Predictor, seqlen: int):
    # Try some different ways we could generate sequences in an attempt to find patterns
    # that may be more or less optimized by folding algorithms
    test_seqs = [
        'A' * seqlen,
        'G' * (seqlen // 2) + 'C' * (seqlen // 2) + 'C' * (seqlen % 2),
        'AUG' * (seqlen // 3) + 'A' * (seqlen % 3),
        ''.join('AUGC'[random.randint(0, 3)] for _ in range(seqlen)),
        ''.join('AUGC'[random.randint(0, 3)] * 3 for _ in range(seqlen//3)) + 'A' *  (seqlen % 3),
    ]
    s = ''
    while (m := seqlen - len(s)) > 0:
        s += 'AUGC'[random.randint(0,3)] * random.randint(1, min(m, 6))
    test_seqs.append(s)

    # Same but for reactivity
    test_reactivities = [
        [None for _ in range(seqlen)],
        [0 for _ in range(seqlen)],
        [0.5 for _ in range(seqlen)],
        [1 for _ in range(seqlen)],
        [random.random() for _ in range(seqlen)],
        list(itertools.chain.from_iterable([[0,1] for _ in range(seqlen // 2)])) + [0] * (seqlen % 2),
    ] if predictor.uses_experimental_reactivities else [[None for _ in range(seqlen)]]

    test_cases = list(itertools.product(test_seqs, test_reactivities))
    # Run each one a few times to factor in noise
    test_cases *= 3


    samples = [sample_in_process(lambda: predictor.run(seq, reactive), MAX_TIMEOUT, predictor.gpu) for (seq, reactive) in test_cases]

    samples = [
        sample for sample in samples
        # If some parts returned all x, this means arnie encountered a failure. We dont want to include those
        # samples as they may have exited earlier than expected (and we consider those situations outliers
        # which we don't want to model)
        if not (sample.result is None or any(all(char == 'x' for char in struct) for struct in sample.result.values()))
    ]

    # If all of them failed, this probably means we OOMed, which we want to treat the same as a timeout
    if len(samples) == 0:
        raise FullSampleFailure()

    return AggregateResourceSample(
        seqlen,
        min(sample.time for sample in samples),
        sum(sample.time for sample in samples) / len(samples),
        max(sample.time for sample in samples),
        max(sample.maxrss for sample in samples),
        max(sample.maxgpu for sample in samples),
    )

def fit_model(samples: list[AggregateResourceSample], key: Callable[[AggregateResourceSample], int | float]):
    # We attempt to fit polynomial models of multiple degrees
    # While we are using lasso regression already to reduce overfitting, we'll also not choose a model
    # with more parameters if the performance does not substantially improve
    best_model = None
    best_model_score = -math.inf
    for deg in range(0, 8):
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=deg)),
            ('linear', Lasso(fit_intercept=False, positive=True, alpha=0.1))
        ]).fit([[sample.seqlen] for sample in samples], [key(sample) for sample in samples])
        score = model.score([[sample.seqlen] for sample in samples], [key(sample) for sample in samples])
        if score > best_model_score + 0.05:
            best_model = model
            best_model_score = score
    return best_model

def sample_resources(predictor: Predictor):
    samples: list[AggregateResourceSample] = []

    # We increase by powers of 4 to try and quickly hone in on 
    seqlen = 16
    failed_small_seqlen = 0
    failed_large_seqlen = math.inf
    while True:
        print('Sampling at length', seqlen)
        try:
            sample = sample_resources_at_length(predictor, seqlen)
            # We require samples to run for at least one second so that memory has been sampled at least 20 times
            if sample.max_runtime > 1:
                samples.append(sample)
            else:
                failed_small_seqlen = seqlen
        except (TimeoutError, FullSampleFailure):
            failed_large_seqlen = seqlen
        
        # We want to have at least MIN_SAMPLES samples to fit to to ensure a good fit and samples of at least
        # MIN_TIMEOUT duration to limit fitting on noise
        if len(samples) == MIN_SAMPLES and any(sample.max_runtime > MIN_TIMEOUT or sample.seqlen >= MAX_LEN for sample in samples):
            break

        # We've hit our upper bound on what makes sense to benchmark
        if seqlen == MAX_LEN:
            failed_large_seqlen = MAX_LEN
        
        if math.isinf(failed_large_seqlen):
            # We haven't hit our max timeout yet - aggressively increase our sequence length
            # Technically this behavior of using `min` could wind up with two samples very
            # close together (eg worst case if seqlen is MAX_LEN-1), but I find that preferrable
            # to the alternative which is either greatly overshooting or undershooting MAX_LEN
            # for the largest sample
            seqlen = min(seqlen * 4, MAX_LEN)
        else:
            if len(samples) < MIN_SAMPLES:
                # Find the biggest gap between two prior "segments" and add a sample between the two
                # Eg 16/64/256/1024, the biggest gap is 1024-256 so we add at 512. We do this so that
                # we can get the most "signal" out of a sample, preferring under-sampled areas of the
                # growth function (also due to our exponential growth, this also means initially preferring
                # larger runtimes which are less likely to have noise)
                #
                # Note the `if sample.seqlen < failed_seqlen`. It's *possible* that for some reason
                # (eg, due to some sort of resource contension/noise) that we could have succeeded on a longer
                # sequence length but failed on a shorter one. We want to avoid re-trying samples that are
                # of a length >= our shortest failed run. However it is ok to try a sample that is smaller
                # than our smallest fail but larger than the next one smaller, so we include failed_seqlen
                # as our upper bound when determining a valid sample range
                sample_lens = [failed_small_seqlen] + sorted(sample.seqlen for sample in samples if sample.seqlen < failed_large_seqlen) + [failed_large_seqlen]
                max_diff = 0
                seqlen_range = (0, 0)
                for (low, high) in itertools.pairwise(sample_lens):
                    if high - low > max_diff:
                        max_diff = high - low
                        seqlen_range = (low, high)
                # This would presumably mean lengths of 1 through MIN_SAMPLES all timed out, but will include
                # the sanity check just in case so we don't run into an infinite loop
                if abs(seqlen_range[0] - seqlen_range[1]) < 2:
                    print(f'Couldn\'t find valid sequence length to sample, bailing early - current samples (n={len(samples)}, max_runtime={max(sample.max_runtime for sample in samples)}): {samples}')
                    return samples

                seqlen = (seqlen_range[0] + seqlen_range[1]) // 2
            else:
                # We already have enough samples, we just want to try and try and get a longer sample to hit MIN_RUNTIME.
                # Split the difference between the highest successful and lowest failed length to continue our binary search
                max_success = max(sample.seqlen for sample in samples)
                # As noted in the condition above, it's possible our longest successful sequence is larger than our smallest failed
                # sequence eg due to noise. Whether that happens or we don't have any more samples between the two values,
                # bail as we have nothing else worth checking
                if max_success > failed_large_seqlen or failed_large_seqlen - max_success < 2:
                    print(f'Couldn\'t find valid sequence length to sample, bailing early - current samples (n={len(samples)}, max_runtime={max(sample.max_runtime for sample in samples)}): {samples}')
                    return samples
                seqlen = (failed_large_seqlen + max_success) // 2

    return samples

def model_resource_usage(predictor: Predictor):
    samples = sample_resources(predictor)
    return {
        'max_runtime': fit_model(samples, lambda sample: sample.max_runtime),
        'avg_runtime': fit_model(samples, lambda sample: sample.average_runtime),
        'min_runtime': fit_model(samples, lambda sample: sample.min_runtime),
        'max_memory': fit_model(samples, lambda sample: sample.max_memory),
        'max_gpu': fit_model(samples, lambda sample: sample.max_gpu),
    }

MODEL_TIMEOUT = (
    MAX_TIMEOUT
    * MIN_SAMPLES
    # Number of different test sequences
    * 6
    # Number of different test reactivities
    * 6
)

def generate_predictor_resource_model(predictor: Predictor):
    models = {
        k: ' + '.join(
            f'{float(coef)}*x**{i}' for i, coef in enumerate(model.named_steps['linear'].coef_)
        )
        for (k, model) in model_resource_usage(predictor).items()
    }
    print(f'Resource usage models for {predictor}: {models}')
