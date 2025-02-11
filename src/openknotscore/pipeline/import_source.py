from typing import TypedDict, Any
import math
import glob
import itertools
import pandas as pd
import rdat_kit
from arnie.utils import convert_bp_list_to_dotbracket, convert_dotbracket_to_bp_list

SourceDef = str | TypedDict('SourceDef', {'path': str, 'extensions': dict[str, Any]})

def load_sources(source_defs: SourceDef | list[SourceDef]) -> pd.DataFrame:
    '''
    Reads source files into a single dataframe

    Right now we only support rdat, but we can expand to support other file types
    (csv, parquet, pickle, etc) in the future
    '''

    source_defs = source_defs if type(source_defs) == list else [source_defs]
    source_defs = [source if type(source) == dict else {'path': source, 'extensions': {}} for source in source_defs]

    dfs = []
    for source in source_defs:
        for source_file in glob.glob(source['path']):
            if not source_file.lower().endswith('rdat'):
                raise ValueError(f'Invalid file extension for source file {source} - only rdat is supported')
            df = load_rdat(source_file)
            for (k, v) in source['extensions'].items():
                df[k] = v
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def load_extension_sources(source_globs: str | list[str], df: pd.DataFrame) -> pd.DataFrame:
    '''
    Reads source files and merges into an existing dataframe
    '''

    _source_globs = [source_globs] if type(source_globs) == str else source_globs
    source_files = list(itertools.chain.from_iterable(glob.glob(source) for source in _source_globs))

    for source in source_files:
        if not source.lower().endswith('csv'):
            raise ValueError(f'Invalid file extension for source file {source} - only csv is supported')

    for source in source_files:
        extension = load_csv(source)
        df.update(
            df[['eterna_id']].merge(extension, on=['eterna_id'], how='left')
        )
    
    return df

def load_rdat(source_file: str):
    rdat = rdat_kit.RDATFile()
    with open(source_file, 'r') as f:
        rdat.load(f)

    modifier = rdat.annotations.get('modifier')
    if modifier:
        if len(modifier) > 1:
            raise Exception('RDAT contained multiple modifier annotations - if this was expected, we need to change our input and output handling to accomodate')
        modifier = modifier[0]
    chemical = rdat.annotations.get('chemical')
    temperature = rdat.annotations.get('temperature')
    if temperature:
        if len(temperature) > 1:
            raise Exception('RDAT contained multiple temperature annotations - if this was expected, we need to change our input and output handling to accomodate')
    temperature = temperature[0]

    for construct in rdat.constructs.values():
        seqList = []

        BLANK_OUT5, BLANK_OUT3 = get_global_blank_out(construct)

        # Loop through all sequences in the RDAT, extract relevant data, and add to the dataframe
        for sequence in construct.data:
            # Grab annotations
            eterna_id = None
            title = None
            score_start_idx = None
            score_end_idx = None
            if 'Eterna' in sequence.annotations:
                for annot in sequence.annotations.get('Eterna'):
                    if annot.startswith('id:'):
                        eterna_id = annot.removeprefix('id:')
                    if annot.startswith('design_name:'):
                        title = annot.removeprefix('design_name:')
                    if annot.startswith('score:start_idx:'):
                        score_start_idx = int(annot.removeprefix('score:start_idx:'))
                    if annot.startswith('score:end_idx:'):
                        score_end_idx = int(annot.removeprefix('score:end_idx:'))
            if title is None:
                title = sequence.annotations.get('name')[0]
            seq = sequence.annotations.get('sequence')[0]
            reads = None
            if 'reads' in sequence.annotations:
                reads = int(sequence.annotations.get('reads')[0])
            signal_to_noise = sequence.annotations.get('signal_to_noise')[0]
            snr = float(sequence.annotations.get('signal_to_noise')[0].split(":")[1])
            warning = sequence.annotations.get('warning', '-')[0]
            if score_start_idx is None:
                score_start_idx = BLANK_OUT5 + 1
            if score_end_idx is None:
                score_end_idx = len(seq) - BLANK_OUT3
            structure = None
            if 'structure' in sequence.annotations:
                structure = sequence.annotations.get('structure')[0] or None

            if structure != None:
                if len(structure) != len(seq):
                    print('Invalid target structure - length mismatch', structure)
                    structure = None
                else:
                    try:
                        convert_bp_list_to_dotbracket(convert_dotbracket_to_bp_list(structure, allow_pseudoknots=True), len(structure))
                    except Exception as e:
                        print(f'Invalid target structure ({e}): {structure}')
                        structure = None

            # Get reactivity data and errors
            reactivity = [None]*BLANK_OUT5 + [val if not math.isnan(val) else None for val in sequence.values] + [None]*BLANK_OUT3
            errors = [None]*BLANK_OUT5 + [val if not math.isnan(val) else None for val in sequence.errors] + [None]*BLANK_OUT3

            # Create a dataframe from this row
            row = {
                'eterna_id': eterna_id,
                'title': title,
                'sequence': seq,
                'reads': reads,
                'signal_to_noise': signal_to_noise,
                'snr': snr,
                'warning': warning,
                'reactivity': reactivity,
                'errors': errors,
                'modifier': modifier,
                'checmical': chemical,
                'temperature': temperature,
                'target_structure': structure,
                'score_start_idx': score_start_idx,
                'score_end_idx': score_end_idx,
            }

            seqList.append(row)

    return pd.DataFrame(seqList)

def get_global_blank_out(construct):
    '''
    Determines the number of bases on either end of the sequence which have no reactivity/error values
    defined (ie, to make the length of the values and errors arrays from the RDATFile match the length
    of the sequence, you need to pad on the left with BLANK_OUT5 nan values and the right with BLANK_OUT3 nan values)
    '''

    assert construct.seqpos == list(range(construct.seqpos[0], construct.seqpos[-1] + 1)), 'RDAT reactivity indicies (SEQPOS) are not contiguous'
    seqlens = [len(seq.annotations['sequence']) for seq in construct.data]
    assert all(seqlen == seqlens[0] for seqlen in seqlens), 'Not all sequences are the same length'

    BLANK_OUT5 = construct.seqpos[0] - construct.offset - 1
    BLANK_OUT3 = len(construct.sequence) - (construct.seqpos[-1] - construct.offset)

    return (BLANK_OUT5, BLANK_OUT3)

def load_csv(source_file: str):
    df = pd.read_csv(source_file)
    df = df[[col for col in df.columns if col in ['eterna_id', 'score_start_idx', 'score_end_idx']]]
    df['eterna_id'] = df['eterna_id'].astype(str)

    return df
