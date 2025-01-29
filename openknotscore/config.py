from typing import Literal, Callable
from abc import ABC, abstractmethod
import itertools
import glob
import tempfile
import os
from os import path
import math
import pandas as pd
import rdat_kit
from .pipeline.prediction import predictors
from .pipeline.import_source import get_global_blank_out
from .substation.runners.runner import Runner
from .substation.runners.local import LocalRunner

class OutputConfig:
    @abstractmethod
    def write(self, df: pd.DataFrame, config: 'OKSPConfig'):
        pass

class RDATOutput(OutputConfig):
    def __init__(self, output_dir: str, max_per_rdat: int = None):
        self.output_dir = output_dir
        self.max_per_rdat = max_per_rdat

    def write(self, df: pd.DataFrame, config: 'OKSPConfig'):
        os.makedirs(self.output_dir, exist_ok=True)

        source_globs = config.source_files
        _source_globs = [source_globs] if type(source_globs) == str else source_globs
        source_files = list(itertools.chain.from_iterable(glob.glob(source) for source in _source_globs))

        for source in source_files:
            if not source.lower().endswith('rdat'):
                raise ValueError(f'Cant write rdat for {source} - input needs to be rdat')

        for source in source_files:
            rdat = rdat_kit.RDATFile()
            with open(source, 'r') as f:
                rdat.load(f)
            for construct in rdat.constructs.values():
                for sequence in construct.data:
                    BLANK_OUT5, BLANK_OUT3 = get_global_blank_out(construct)

                    seq = sequence.annotations["sequence"][0]
                    annotationList = sequence.annotations.setdefault('Eterna', [])
                    
                    # There may be no processed data (OKS, predictions) associated with the sequence
                    # if the sequence had low-quality or missing reactivity data, so we skip those rows
                    row = df[df['sequence'] == seq]
                    row = row[row['reactivity'].apply(lambda x: x==[None]*BLANK_OUT5 + sequence.values + [None]*BLANK_OUT3)]
                    row = row.squeeze()
                    if row.empty:
                        continue

                    # Add annotations with the processed data to the RDAT
                    annotationList.append(f"score:openknot_score:{row['ensemble_OKS']:.6f}")
                    annotationList.append(f"score:eterna_classic_score:{row['ensemble_ECS']:.6f}")
                    annotationList.append(f"score:crossed_pair_quality_score:{row['ensemble_CPQ']:.6f}")
                    annotationList.append(f"best_fit:tags:{','.join(row['ensemble_tags'])}")
                    annotationList.append(f"best_fit:structures:{','.join(row['ensemble_structures'])}")
                    annotationList.append(f"best_fit:eterna_classic_scores:{','.join([f'{v:.6f}' for v in row['ensemble_structures_ecs']])}")
            
            tempout = tempfile.NamedTemporaryFile(delete=False)
            tempout.close()
            rdat.save(tempout.name)

            if self.max_per_rdat is None:
                with open(tempout.name) as rf:
                    with open(f'{self.output_dir}/{path.basename(source)}', 'w') as wf:
                        wf.write(rf.read())
            else:
                num_splits = math.ceil(len(list(rdat.constructs.values())[0].data)/self.max_per_rdat)

                for i in range(num_splits):
                    # Reset the RDAT object
                    rdat = rdat_kit.RDATFile()
                    with open(tempout.name, 'r') as f:
                        rdat.load(f)

                    for (construct_name, construct) in rdat.constructs.items():
                        data = rdat.constructs[construct_name].data
                        # Subset the RDAT data for saving.
                        rdat.constructs[construct_name].data = data[i*self.max_per_rdat:(i+1)*self.max_per_rdat]
                        rdat.values[construct_name] = [row.values for row in data[i*self.max_per_rdat:(i+1)*self.max_per_rdat]]
                        rdat.errors[construct_name] = [row.errors for row in data[i*self.max_per_rdat:(i+1)*self.max_per_rdat]]

                    rdat.save(f'{self.output_dir}/{path.basename(source).removesuffix('.rdat')}-{i+1}.rdat')
            
            os.unlink(tempout.name)

class CSVOutput(OutputConfig):
    def __init__(self, output_path: str, sep: str=',', compression: Literal['snappy', 'gzip', 'brotli', 'lz4', 'zstd'] | None = None):
        self.output_path = output_path
        self.sep=sep
        self.compression = compression

    def write(self, df: pd.DataFrame, config: 'OKSPConfig'):
        os.makedirs(path.dirname(self.output_path), exist_ok=True)
        df.to_csv(self.output_path, sep=self.sep, compression=self.compression, index=False)

class ParquetOutput(OutputConfig):
    def __init__(self, output_path: str, compression: Literal['snappy', 'gzip', 'brotli', 'lz4', 'zstd'] | None = None):
        self.output_path = output_path
        self.compression = compression

    def write(self, df: pd.DataFrame, config: 'OKSPConfig'):
        os.makedirs(path.dirname(self.output_path), exist_ok=True)
        df.to_parquet(self.output_path, compression=self.compression)

class OutputFilter(OutputConfig):
    def __init__(self, filter: Callable[[pd.DataFrame], pd.DataFrame], outputs: list[OutputConfig]):
        self.filter = filter
        self.outputs = outputs

    def write(self, df: pd.DataFrame, config: 'OKSPConfig'):
        filtered = self.filter(df)
        for out in self.outputs:
            out.write(filtered, config)

class OKSPConfig(ABC):
    @property
    @abstractmethod
    def db_path() -> str | list[str]:
        '''
        Path to a directory that can be used to store internal files (eg, caches, intermediate outputs, etc)
        '''
        pass

    @property
    @abstractmethod
    def source_files() -> str | list[str]:
        '''
        A single or list of file paths or globs that should be loaded as input data to the pipeline
        '''
        pass

    extension_source_files: str | list[str] = []
    '''
    A single or list of file paths or globs that should be loaded as input data to the pipeline not as
    individual solutions to be run through the pipeline, but additional data to extend data loaded from
    source_files. Namely, columns will be merged with eterna_id being the joining column
    '''

    @staticmethod
    def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
        '''
        If you don't want to run computations for all data in source_files, use this function to
        filter down to just the rows you want to compute for
        '''
        return df

    @property
    @abstractmethod
    def output_configs() -> list[OutputConfig]:
        pass

    imported_column_map: dict[str, str] = {}
    
    enabled_predictors: list[predictors.Predictor] = [
        predictors.Vienna2Predictor(as_name='vienna2'),
        predictors.Contrafold2Predictor(as_name='contrafold2'),
        predictors.EternafoldPredictor(as_name='eternafold'),
        predictors.RnastructurePredictor(as_name='rnastructure_mfe'),
        # Takes too long
        #predictors.E2efoldPredictor(as_name='e2efold'),
        predictors.HotknotsPredictor(as_name='hotknots'),
        predictors.IpknotPredictor(as_name='ipknot'),
        predictors.KnottyPredictor(as_name='knotty'),
        predictors.PknotsPredictor(as_name='pknots'),
        predictors.SpotrnaPredictor(as_name='spotrna'),
        # Never got this working
        #predictors.Spotrna2Predictor(as_name='spotrna2')
        predictors.ShapifyHfoldPredictor(as_name='shapify-hfold'),
        predictors.NupackPkPredictor(as_name='nupack_pk'),
        predictors.RnastructureShapePredictor(as_name='rnastructure+SHAPE'),
        predictors.ShapeknotsPredictor(as_name='shapeknots'),
        predictors.Vienna2PkFromBppPredictor()
            .add_heuristic('threshknot', as_name='vienna_2.TK')
            .add_heuristic('hungarian', as_name='vienna_2.HN'),
        predictors.EternafoldPkFromBppPredictor()
            .add_heuristic('threshknot', as_name='eternafold.TK')
            .add_heuristic('hungarian', as_name='eternafold.HN'),
        predictors.Contrafold2PkFromBppPredictor()
            .add_heuristic('threshknot', as_name='contrafold_2.TK')
            .add_heuristic('hungarian', as_name='contrafold_2.HN'),
        predictors.RibonanzaNetSSPredictor('ribonanzanet-ss'),
        predictors.RibonanzaNetShapeDerivedPredictor()
            .add_configuration('2a3', 'rnastructure', as_name='rnet-2a3-rnastructure')
            .add_configuration('dms', 'rnastructure', as_name='rnet-dms-rnastructure')
            .add_configuration('2a3', 'shapeknots', as_name='rnet-2a3-shapeknots')
            .add_configuration('dms', 'shapeknots', as_name='rnet-dms-shapeknots')
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

    filter_singlets = False
    '''
    Whether to filter out singlet base pairs (stacks/helices which only contain one pair)
    '''
    
