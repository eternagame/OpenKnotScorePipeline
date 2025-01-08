import sys
from os import path
sys.path.append(path.join(path.dirname(__file__), '../..'))

from openknotscore.cli import OKSPConfig, run_cli

class Config(OKSPConfig):
    source_files = path.join(path.dirname(__file__), 'source_rdats/*')
    db_path = path.join(path.dirname(__file__), 'db')
