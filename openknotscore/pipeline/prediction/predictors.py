from abc import ABC, abstractmethod
import subprocess
import re
import math
from arnie.utils import convert_dotbracket_to_bp_list, convert_bp_list_to_dotbracket
try:
    from arnie.mfe import mfe
    from arnie.bpps import bpps
    from arnie.pk_predictors import pk_predict, pk_predict_from_bpp
except Exception as e:
    # Arnie unfortunately raises an exception on import if no predictors are configured
    # If the user isn't using arnie, we don't want to error, so we'll defer raising an error
    # to when attempting to use these methods
    def arnie_load_failed(*args, **kwargs):
        raise e
    mfe = bpps = pk_predict = pk_predict_from_bpp = arnie_load_failed

class Predictor(ABC):
    @property
    @abstractmethod
    def uses_experimental_reactivities() -> bool:
        '''
        Whether or not this predictor takes into account the experimental reactivities in
        deriving its prediction. If it does not, we can use cached values for this sequence
        even if the reactivities changed (and also we can run it even if experimental data
        is not available)
        '''
        pass

    @property
    @abstractmethod
    def prediction_names(self) -> list[str]:
        pass

    @abstractmethod
    def run(self, seq: str, reactivities: list[float]) -> dict[str, str]:
        '''
        Run the given predictor, returning a map of prediction names to dot-bracket
        structures. We allow multiple predictions to be returned in case multiple different
        settings can be more efficiently computed in parallel (eg, computing bpps once and then
        running both hungarian and threshknot)
        '''
        pass

    @abstractmethod
    def approximate_min_runtime(self, seq: str) -> int:
        '''
        Returns the approximate lower-bound runtime for computing the prediction for the given
        sequence, in seconds
        
        Values for this are likely computed via openknotscore.pipeline.prediction.sample.model_resource_usage
        '''
        pass

    @abstractmethod
    def approximate_avg_runtime(self, seq: str) -> int:
        '''
        Returns the approximate average-bound runtime for computing the prediction for the given
        sequence, in seconds
        
        Values for this are likely computed via openknotscore.pipeline.prediction.sample.model_resource_usage
        '''
        pass

    @abstractmethod
    def approximate_max_runtime(self, seq: str) -> int:
        '''
        Returns the approximate upper-bound runtime for computing the prediction for the given
        sequence, in seconds
        
        Values for this are likely computed via openknotscore.pipeline.prediction.sample.model_resource_usage
        '''
        pass

    @abstractmethod
    def approximate_max_memory(self, seq: str) -> int:
        '''
        Returns the approximate upper-bound memory for computing the prediction for the given
        sequence, in megabytes
    
        Values for this are likely computed via openknotscore.pipeline.prediction.sample.model_resource_usage
        '''
        pass

    cpus: int = 1

    gpu = False
    def approximate_max_gpu_memory(self, seq: str) -> int:
        '''
        Returns the approximate upper-bound GPU memory for computing the prediction for the given
        sequence, in megabytes
        '''
        return 0

class ArnieMfePredictor(Predictor):
    uses_experimental_reactivities = False
    prediction_names = []

    def __init__(self, package_name: str, as_name: str, arnie_kwargs: dict = None):
        self.name = as_name
        self.prediction_names = [self.name]
        self.package_name = package_name
        self.arnie_kwargs = arnie_kwargs or {}

    def run(self, seq: str):
        return {
            self.name: mfe(seq, self.package_name, **self.arnie_kwargs)
        }

class ArnieMfeShapePredictor(ArnieMfePredictor):
    uses_experimental_reactivities = True

    def run(self, seq: str, reactivities: list[float]):
        # arnie expects a list of float values; we need to account for the leader region with invalid data,
        # so we don't provide the reactivity data directly stored in the row
        processed_reactivities = [value if not math.isnan(value) else 0 for value in reactivities]
        return {
            self.name: mfe(seq, self.package_name, shape_signal=processed_reactivities, **self.arnie_kwargs)
        }
    
class ArniePkPredictor(Predictor):
    uses_experimental_reactivities = False
    prediction_names = []

    def __init__(self, package_name: str, as_name: str, arnie_kwargs: dict = None):
        self.name = as_name
        self.prediction_names = [self.name]
        self.package_name = package_name
        self.arnie_kwargs = arnie_kwargs or {}

    def run(self, seq: str, reactivities: list[float]):
        return {
            self.name: pk_predict(seq, self.package_name, **self.arnie_kwargs)
        }

class ArniePkFromBppPredictor(Predictor):
    uses_experimental_reactivities = False
    prediction_names = []

    def __init__(self, package_name: str, arnie_bpp_kwargs: dict = None):
        self.package_name = package_name
        self.arnie_bpp_kwargs = arnie_bpp_kwargs or {}
        self.heuristics = []

    def add_heuristic(self, heuristic, as_name: str, arnie_heuristic_kwargs: dict = None):
        self.prediction_names.append(as_name)
        self.heuristics.append({
            'heuristic': heuristic,
            'arnie_heuristic_kwargs': arnie_heuristic_kwargs,
            'name': as_name
        })
        return self

    def run(self, seq: str, reactivities: list[float]):
        # TODO: Figure out how to set things up such that we only compute bpps once for all heuristics
        bpp = bpps(seq, package=self.package_name, **self.arnie_bpp_kwargs)
        return {
            heuristic['name']: pk_predict_from_bpp(bpp, heuristic=heuristic['heuristic'], **heuristic['arnie_heuristic_kwargs'])
            for heuristic in self.heuristics
        }

class Vienna2Predictor(ArnieMfePredictor):
    def __init__(self, as_name: str, arnie_kwargs: dict = None):
        super().__init__('vienna_2', as_name, arnie_kwargs)

class Contrafold2Predictor(ArnieMfePredictor):
    def __init__(self, as_name: str, arnie_kwargs: dict = None):
        super().__init__('contrafold_2', as_name, arnie_kwargs)

class EternafoldPredictor(ArnieMfePredictor):
    def __init__(self, as_name: str, arnie_kwargs: dict = None):
        super().__init__('eternafold', as_name, arnie_kwargs)

class RnastructurePredictor(ArnieMfePredictor):
    def __init__(self, as_name: str, arnie_kwargs: dict = None):
        super().__init__('rnastructure', as_name, arnie_kwargs)

class E2efoldPredictor(ArniePkPredictor):
    def __init__(self, as_name: str, arnie_kwargs: dict = None):
        super().__init__('e2efold', as_name, arnie_kwargs)

class HotknotsPredictor(ArniePkPredictor):
    def __init__(self, as_name: str, arnie_kwargs: dict = None):
        super().__init__('hotknots', as_name, arnie_kwargs)

class IpknotsPredictor(ArniePkPredictor):
    def __init__(self, as_name: str, arnie_kwargs: dict = None):
        super().__init__('ipknots', as_name, arnie_kwargs)

class KnottyPredictor(ArniePkPredictor):
    def __init__(self, as_name: str, arnie_kwargs: dict = None):
        super().__init__('knotty', as_name, arnie_kwargs)

class PknotsPredictor(ArniePkPredictor):
    def __init__(self, as_name: str, arnie_kwargs: dict = None):
        super().__init__('pknots', as_name, arnie_kwargs)

class SpotrnaPredictor(ArniePkPredictor):
    def __init__(self, as_name: str, arnie_kwargs: dict = None):
        super().__init__('spotrna', as_name, arnie_kwargs)

class Spotrna2Predictor(ArniePkPredictor):
    def __init__(self, as_name: str, arnie_kwargs: dict = None):
        super().__init__('spotrna2', as_name, arnie_kwargs)

class NupackPkPredictor(ArniePkPredictor):
    def __init__(self, as_name: str, arnie_kwargs: dict = None):
        super().__init__('nupack', as_name, arnie_kwargs)

class RnastructureShapePredictor(ArnieMfeShapePredictor):
    def __init__(self, as_name: str, arnie_kwargs: dict = None):
        super().__init__('rnastructure', as_name, arnie_kwargs)

class ShapeknotsPredictor(ArnieMfeShapePredictor):
    def __init__(self, as_name: str, arnie_kwargs: dict = None):
        super().__init__('rnastructure', as_name, {
            **arnie_kwargs,
            'pseudo': True
        })

class Vienna2PkFromBppPredictor(ArniePkFromBppPredictor):
    def __init__(self, arnie_bpp_kwargs: dict = None):
        super().__init__('vienna_2', arnie_bpp_kwargs)

class EternafoldPkFromBppPredictor(ArniePkFromBppPredictor):
    def __init__(self, arnie_bpp_kwargs: dict = None):
        super().__init__('eternafold', arnie_bpp_kwargs)

class Contrafold2PkFromBppPredictor(ArniePkFromBppPredictor):
    def __init__(self, arnie_bpp_kwargs: dict = None):
        super().__init__('contrafold_2', arnie_bpp_kwargs)

# TODO: Move into Arnie
class ShapifyHfoldPredictor(Predictor):
    uses_experimental_reactivities = False
    name = ''

    def __init__(self, as_name: str, hfold_location: str):
        self.hfold_location = hfold_location
        self.name = as_name
        self.prediction_names = [as_name]

    def run(self, seq: str):
        hfold_return = subprocess.run([self.hfold_location+"/HFold_iterative", "--s", seq.strip()], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")

        # Pull structure out of Shapify output
        match = re.search(r"Result_0: (.*) \n", hfold_return.stdout)
        structure = match.group(1) if match else "ERR"
        
        # Sanitize dbn structure to match our conventions 
        return {
            self.name: convert_bp_list_to_dotbracket(convert_dotbracket_to_bp_list(structure, allow_pseudoknots=True), len(structure))
        }
