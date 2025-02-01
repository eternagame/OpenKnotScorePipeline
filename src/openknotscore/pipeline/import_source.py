import glob
import itertools
import pandas as pd
import rdat_kit

def load_sources(source_globs: str | list[str]) -> pd.DataFrame:
    '''
    Reads source files into a single dataframe

    Right now we only support rdat, but we can expand to support other file types
    (csv, parquet, pickle, etc) in the future
    '''

    _source_globs = [source_globs] if type(source_globs) == str else source_globs
    source_files = list(itertools.chain.from_iterable(glob.glob(source) for source in _source_globs))

    for source in source_files:
        if not source.lower().endswith('rdat'):
            raise ValueError(f'Invalid file extension for source file {source} - only rdat is supported')
    
    dfs = []
    for source in source_files:
        dfs.append(load_rdat(source))
    return pd.concat(dfs, ignore_index=True)

def load_rdat(source_file: str):
    rdat = rdat_kit.RDATFile()
    with open(source_file, 'r') as f:
        rdat.load(f)

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

            # Get reactivity data and errors
            reactivity = [None]*BLANK_OUT5 + sequence.values + [None]*BLANK_OUT3
            errors = [None]*BLANK_OUT5 + sequence.errors + [None]*BLANK_OUT3

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
