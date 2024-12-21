from dataclasses import dataclass
import pandas as pd
from .predictors import Predictor
from ...plan.domain import Task, ResourceConfiguration

@dataclass
class PredictionTaskInfo:
    predictor: Predictor
    row: pd.Series

def get_predict_runtime(predictor: Predictor, row: pd.Series):
    def predict_runtime(resources: ResourceConfiguration):
        return predictor.approximate_runtime(row['sequence'], resources)
    return predict_runtime

def get_predict_mem(predictor: Predictor, row: pd.Series):
    def predict_mem(resources: ResourceConfiguration):
        return predictor.approximate_memory(row['sequence'], resources)
    return predict_mem

def get_predict_gpu_mem(predictor: Predictor, row: pd.Series):
    def predict_gpu_mem(resources: ResourceConfiguration):
        return predictor.approximate_gpu_memory(row['sequence'])
    return predict_gpu_mem

def get_prediction_task(predictor: Predictor, row: pd.Series):
    return Task[PredictionTaskInfo](
        PredictionTaskInfo(predictor, row),
        get_predict_runtime(predictor, row),
        get_predict_mem(predictor, row),
        get_predict_gpu_mem(predictor, row),
    )