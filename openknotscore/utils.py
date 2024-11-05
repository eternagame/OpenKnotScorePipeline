import rdat_kit

def load_rdat(file_path):
    rdat = rdat_kit.RDATFile()
    with open(file_path, 'r') as f:
        rdat.load(f)

    assert len(rdat.constructs) == 1
    construct_name = list(rdat.constructs.keys())[0]

    return (rdat, construct_name)

def get_global_blank_out(construct):
    assert construct.seqpos == list(range(construct.seqpos[0], construct.seqpos[-1] + 1)), 'RDAT reactivity indicies (SEQPOS) are not contiguous'
    seqlens = [len(seq.annotations['sequence']) for seq in construct.data]
    assert all(seqlen == seqlens[0] for seqlen in seqlens), 'Not all sequences are the same length'

    BLANK_OUT5 = construct.seqpos[0] - construct.offset - 1
    BLANK_OUT3 = len(construct.sequence) - (construct.seqpos[-1] - construct.offset)

    return (BLANK_OUT5, BLANK_OUT3)
