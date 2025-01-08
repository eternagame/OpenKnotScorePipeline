from dataclasses import dataclass
import pandas as pd
from .predictors import Predictor
from ...plan.domain import Task, Runnable, AccessibleResources, UtilizedResources

@dataclass
class ResourcePredictor:
    predictor: Predictor
    row: pd.Series

    def predict(self, resources: AccessibleResources):
        return UtilizedResources(
            self.predictor.approximate_max_runtime(self.row['sequence'], resources),
            self.predictor.approximate_avg_runtime(self.row['sequence'], resources),
            self.predictor.approximate_min_runtime(self.row['sequence'], resources),
            self.predictor.approximate_max_memory(self.row['sequence'], resources),
            self.predictor.approximate_max_gpu_memory(self.row['sequence']),
        )

def get_prediction_task(predictor: Predictor, row: pd.Series):
    return Task(
        Runnable.create(predictor.run)(row['sequence'], row.get('reactivity')),
        ResourcePredictor(predictor, row).predict,
        predictor.requires_gpu
    )
