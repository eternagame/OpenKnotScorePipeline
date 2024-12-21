import pandas as pd
import sys
from os import path
sys.path.append(path.join(path.dirname(__file__), '../..'))

from openknotscore.cli import OKSPConfig, run_cli

class Config(OKSPConfig):
    source_files = path.join(path.dirname(__file__), 'source_rdats/*')
    db_path = path.join(path.dirname(__file__), 'db')

    def filter_for_computation(df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[df['warning'] != 'badQuality']

if __name__ == '__main__':
    run_cli(Config)
