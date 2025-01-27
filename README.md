# OpenKnotScorePipeline
Python pipeline to generate OpenKnot Scores for Eterna sequence libraries

## Setup
Python >= 3.12 is required

You will also need to set up prediction packages according to [Arnie's conventions](https://daslab.github.io/arnie/#/setup/environment)
for packages supported via arnie, and for other packages:
* Set the `HFOLD_PATH` environment variable to the directory for shapify-hfold
* Set the `RIBONANZANET_PATH` environment variable to the directory with the RibonanzaNet source code
* Set the `RIBONANZANET_WEIGHTS` environment variable to the directory with the RibonanzaNet pretrained weights
* Set the `RIBONANZANET_ENV_PATH` environment variable to the bin directory for the virtual environment
  (or conda/mabma environment) for RibonanzaNet

Also, set `CONTRAFOLD_2_PARAMS_PATH` to the location of the contrafold 2 parameter file (we run Eternafold
instead of contrafold in order to use the latest fixes eg avoiding <3 nt loops, but with contrafold parameters)

## Full pipeline runs
The recommended setup is as follows:

Create a directory for your pipeline run, such as one of those in `examples/`. Create a python file
defining a subclass of `openknotscore.config.OKSPConfig` called `Config`, filling out the parameters
as necessary (refer to the examples or the docstrings for how this can be configured).

```
python -m venv .venv
source .venv/bin/activate
pip install https://github.com/eternagame/OpenKnotScorePipeline
pip freeze > frozen-requirements.txt
```
This ensures the exact set of dependencies are recorded for auditing and reproducibility. If you later
need to re-install dependencies (ie in a fresh venv), you can `pip install -r frozen-requirements.txt`.
If you upgrade dependencies, you should make sure you `pip freeze > frozen-requirements.txt` again. 

We recommend doing this once per pipeline to ensure that you know what version of the pipeline
a given set of artifacts (including the intermediate database files) originated from.

Additionally for auditing and reproducibility you may want to somehow version the external predictor
dependencies. You might do this by setting up a container, by creating "immutable" versioned directories
containing the predictors, or tar them and extract as a preperatory step for pipeline runs
(eg using the `init_script` parameter for `SlurmRunner` when configuring the runner in your config).

## As a dependency
If you need to use this library as a dependency, such as just to access the scoring functions,
you can do so via `pip install https://github.com/eternagame/OpenKnotScorePipeline`

## For development
After cloing this repository, you can run `pip install .`. You could then run projects in the `examples/`
directory or create new projects under `data/`

## How to use this pipeline
For a full list of available commands, run `python -m openknotscore.cli <projectfile> --help`
(for example: `python -m openknotscore.cli examples.pk50-twist.pipeline --help`)

In most cases, you can run `python -m openknotscore.cli <projectfile> predict` to generate predictions,
and then once all predictions have finished, run `-m openknotscore.cli <projectfile> score` to generate
scores and write the output files. Rerunning `predict` will check for any missing predictions (eg
if a predictor was not run or failed for a given row, if a new row was added, if reactivities changed and
the output of a shape-directed predictor is no longer valid) and only rerun those.

If you want to get an idea of the resource requierments/runtime for your configured data and predictors,
you can run `python -m openknotscore.cli <projectfile> predict-forecast`.

If you want to check for missing or failed predictions after running `predict`, 
you can run `python -m openknotscore.cli <projectfile> predict-check-failed`.

If you want to import precomputed predictions from eg a CSV, 
you can run `python -m openknotscore.cli <projectfile> predict-import <path>`.

If you want to remove all predictions for a given predictor from the internal database (eg in the case
there was a bug with its predictions), you can run `python -m openknotscore.cli <projectfile> predict-clear <predictor>`
