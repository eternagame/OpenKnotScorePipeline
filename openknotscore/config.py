from abc import ABC, abstractmethod
import os
import pandas as pd
from .pipeline.prediction import predictors
from .substation.runners.runner import Runner
from .substation.runners.local import LocalRunner

class OKSPConfig(ABC):
    @property
    @abstractmethod
    def source_files() -> str | list[str]:
        '''
        A single or list of file paths or globs that should be loaded as input data to the pipeline
        '''
        pass

    @property
    @abstractmethod
    def db_path() -> str | list[str]:
        '''
        Path to a directory that can be used to store internal files (eg, caches, intermediate outputs, etc)
        '''
        pass

    @staticmethod
    def filter_for_computation(df: pd.DataFrame) -> pd.DataFrame:
        '''
        If you don't want to run computations for all data in source_files, use this function to
        filter down to just the rows you want to compute for
        '''
        return df
    
    enabled_predictors: list[predictors.Predictor] = [
        # TODO: Enable once they fully implement base classes
        # predictors.Vienna2Predictor(as_name='vienna2'),
        # predictors.Contrafold2Predictor(
        #     as_name='contrafold2',
        #     arnie_kwargs={
        #         'params_file': os.environ['CONTRAFOLD_2_PARAMS_PATH']
        #     }
        # ),
        # predictors.EternafoldPredictor(as_name='eternafold'),
        # predictors.RnastructurePredictor(as_name='rnastructure_mfe'),
        # # Takes too long
        # #predictors.E2efoldPredictor(as_name='e2efold'),
        # predictors.HotknotsPredictor(as_name='hotknots'),
        # predictors.IpknotsPredictor(as_name='ipknots'),
        # predictors.KnottyPredictor(as_name='knotty'),
        # predictors.PknotsPredictor(as_name='pknots'),
        # predictors.SpotrnaPredictor(as_name='spotrna'),
        # # Never got this working
        # #predictors.Spotrna2Predictor(as_name='spotrna2')
        # predictors.ShapifyHfoldPredictor(as_name='shapify-hfold', hfold_location=os.environ['HFOLD_PATH']),
        # predictors.NupackPkPredictor(as_name='nupack_pk'),
        # predictors.RnastructureShapePredictor(as_name='rnastructure+SHAPE'),
        # predictors.ShapeknotsPredictor(as_name='shapeknots'),
        # predictors.Vienna2PkFromBppPredictor()
        #     .add_heuristic('threshknots', as_name='vienna_2.TK')
        #     .add_heuristic('hungarian', as_name='vienna_2.HN'),
        # predictors.EternafoldPkFromBppPredictor()
        #     .add_heuristic('threshknots', as_name='eternafold.TK')
        #     .add_heuristic('hungarian', as_name='eternafold.HN'),
        # predictors.Contrafold2PkFromBppPredictor(arnie_bpp_kwargs={
        #         'params_file': os.environ['CONTRAFOLD_2_PARAMS_PATH']
        #     })
        #     .add_heuristic('threshknots', as_name='contrafold_2.TK')
        #     .add_heuristic('hungarian', as_name='contrafold_2.HN')
    ]

    runner: Runner = LocalRunner()

    runtime_buffer = 1.15
    '''
    When calculating the expected max runtime needed to run tasks, use this multiplier to add
    some additional buffer to account for underestimates or variation
    '''

    memory_buffer = 1.15
    '''
    When calculating the expected RAM needed to run tasks, use this multiplier to add
    some additional buffer to account for underestimates or variation
    '''

    gpu_memory_buffer = 1.15
    '''
    When calculating the expected GPU memory needed to run tasks, use this multiplier to add
    some additional buffer to ac
    '''
    
