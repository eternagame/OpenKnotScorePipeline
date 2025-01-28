import sys
from os import path

from openknotscore.config import OKSPConfig, RDATOutput, CSVOutput, ParquetOutput

class Config(OKSPConfig):
    db_path = path.join(path.dirname(__file__), 'db')
    
    source_files = path.join(path.dirname(__file__), 'source_rdats/*')

    output_configs = [
        RDATOutput(path.join(path.dirname(__file__), 'output/rdat')),
        RDATOutput(path.join(path.dirname(__file__), 'output/rdat-eterna'), 100),
        CSVOutput(path.join(path.dirname(__file__), 'output/structures.csv')),
        ParquetOutput(path.join(path.dirname(__file__), 'output/structures.parquet.gz'), 'gzip'),
    ]
