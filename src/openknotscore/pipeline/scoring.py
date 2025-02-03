import re
import pandas
import statistics
from arnie.utils import convert_dotbracket_to_bp_list, post_process_struct

def calculateEternaClassicScore(structure, data, score_start_idx, score_end_idx, filter_singlets=False):
    """Calculates an Eterna score for a predicted structure and accompanying reactivity dataset

    The Eterna Classic score measures how well a structure prediction matches the reactivity data
    
    data: a list of reactivity values, normalized from 0 to 1 (~90th percentile)
    structure: a list of predicted values in dot bracket notation
    score_start_idx: index of data to start scoring from
    score_end_idx: index of data to end scoring at
    """
    # Failed or missing - this is not uncommmon and expected in many cases, so we won't
    # make a lot of noise by printing about it
    if pandas.isna(structure):
        return

    if not isinstance(structure, str):
        print(f"Structure in dbn notation expected, got: {type(structure)}")
        return
    if not isinstance(data, list):
        print(f"List of reactivity values expected, got: {type(data)}")
        print(data)
        return
    if len(structure) != len(data):
        print(f"Structure and data array lengths don't match: {len(structure)} != {len(data)}")
        print(structure)
        return
    
    if filter_singlets:
        structure = post_process_struct(structure, min_len_helix=2)

    # Cutoff values; minimum value threshold of paired or unpaired in SHAPE data
    threshold_SHAPE_fixed = 0.5
    min_SHAPE_fixed = 0.0
    
    # Convert the input structure to a binary paired/unpaired list
    prediction = [1 if char == "." else 0 for char in structure]

    data_scored = data[score_start_idx:score_end_idx+1]
    prediction_scored = prediction[score_start_idx:score_end_idx+1]
    correct_hit = [0] * len(data_scored)
    
    # Loop over each base we are testing for correlation 
    for i in range(len(data_scored)):
        # If the prediction is the base is unpaired
        if (prediction_scored[i] == 1):
        # We check that the data passes a cutoff, and add it to the correct_hit list if true
            if (data_scored[i] > (0.25*threshold_SHAPE_fixed + 0.75*min_SHAPE_fixed)):
                correct_hit[i] = 1
        else:
        # The prediction is the base is paired, we check that the data is below the threshold value
            if (data_scored[i] < threshold_SHAPE_fixed):
                correct_hit[i] = 1
    
    # To get the eterna score, we sum the correct_hit list to figure out how many predicted bases
    # matched the data, then divide by the total number of scored bases and normalize to 100
    eterna_score = ( sum(correct_hit) / len(correct_hit) ) * 100
    return eterna_score

def identify_crossing_bps(bps):
    """Identifies all bases involved in crossing pairs
    
    bps: base pair tuple list ([[11,17],[10,18],[9,19]...])
    Returns a list of indexes of bases involved in crossing pairs
    """
    crossed_res = []
    # Check every base pair in the list
    for bp in bps:
        # The first value in the base pair should be less than the second value
        assert bp[0] < bp[1]
        # Now we loop over the list again, grabbing base pairs to compare
        for next_bp in bps:
            # This checks whether the original bp and the compared bp cross
            # If they do, we add base positions to the crossed residue list
            if (
                bp[0] < next_bp[0] and next_bp[0] < bp[1] and bp[1] < next_bp[1]
            ) or (
                next_bp[0] < bp[0] and bp[0] < next_bp[1] and next_bp[1] < bp[1]
            ):
                crossed_res.append(bp[0])
                crossed_res.append(bp[1])
                crossed_res.append(next_bp[0])
                crossed_res.append(next_bp[1])
                    
    # Returns a unique list, since crossed residues may contain duplicate values
    return list(set(crossed_res)) if len(crossed_res) else []

def remove_bps_in_unscored_regions(bps, score_start_idx, score_end_idx):
    filtered_bps = []
    for bp in bps:
        assert bp[0] < bp[1]
        if (bp[0] < score_start_idx): continue
        if (bp[1] < score_start_idx): continue
        if (bp[0] > score_end_idx): continue
        if (bp[1] > score_end_idx): continue
        filtered_bps.append(bp)
    return filtered_bps

def calculateCrossedPairQualityScore(structure, data, score_start_idx, score_end_idx, filter_singlets=False):
    """Calculates the crossed pair quality score for a DBN structure and accompanying dataset

    The CPQ scores are a pair of metrics that calculate how well the reactivity data
    supports the presence of crossed pairs in a predicted structure. The crossed pair
    score is measured against the entire structure, while the crossed pair quality score
    is measured against only the base pairs that are predicted to be in a crossed pair.
    
    crossed_pair_score = 
    100 * (number of residues in crossed pairs with data < 0.25) /
        [0.7*(length of region with data - 20)]

    crossed_pair_quality_score =
    100 * (number of residues in crossed pairs with data < 0.25) /
        ( number of residues modeled to be crossed pairs in structure )
           
    data: [Nres] data, normalized to go from 0 to 1 (~90th percentile)
    structure: string in dot-bracket notation for structure, with pseudoknots
    score_start_idx: index of data to start scoring from
    score_end_idx: index of data to end scoring at
    """
    # Failed or missing - this is not uncommmon and expected in many cases, so we won't
    # make a lot of noise by printing about it
    if pandas.isna(structure):
        return
    
    if not isinstance(structure, str):
        print(f"Structure in dbn notation expected, got: {type(structure)}")
        return
    if not isinstance(data, list):
        print(f"List of reactivity values expected, got: {type(data)}")
        return
    if len(structure) != len(data):
        print(f"Structure and data array lengths don't match: {len(structure)} != {len(data)}")
        print(structure)
        return
    
    threshold_SHAPE_fixed_pair = 0.25
    
    crossed_pair_score = 0
    crossed_pair_quality_score = 0
    
    # Some algorithms will produce a string of all x's when they fail on a sequence
    # This checks for a structure string that is all x characters and returns if true
    failed_structure = [char == "x" for char in structure]
    if (all(failed_structure)):
            return [crossed_pair_score, crossed_pair_quality_score]

    if filter_singlets:
        # Remove singlet base pairs
        structure = post_process_struct(structure, min_len_helix=2)
    bp_list = convert_dotbracket_to_bp_list(structure, allow_pseudoknots=True)
    
    # Get indexes for bases in crossed pairs
    crossed_res = identify_crossing_bps(bp_list)

    # Filter out base pairs that involve residues in the blanked out flanking regions
    bps_filtered = remove_bps_in_unscored_regions(bp_list, score_start_idx, score_end_idx)
    crossed_res_filtered = identify_crossing_bps(bps_filtered)
    
    num_crossed_pairs  = 0
    crossed_pair_quality_score = 0 
    max_count = 0
    
    for i in crossed_res:
        # Skip if the base index is in a flanking region
        if i < score_start_idx: continue
        if i > score_end_idx: continue
        
        max_count = max_count + 1
        if ( data[i] < threshold_SHAPE_fixed_pair):
            if i in crossed_res_filtered:
                num_crossed_pairs = num_crossed_pairs + 1
            else:
                num_crossed_pairs = num_crossed_pairs + 0.5
                
    data_region_length = 1 + score_end_idx - score_start_idx
    # TODO: Explain heuristic
    max_crossed_pairs = 0.7 * max(data_region_length - 20, 20)
    crossed_pair_score = 100 * min( num_crossed_pairs/max_crossed_pairs, 1.0)

    if max_count > 0:
        crossed_pair_quality_score = 100 * (num_crossed_pairs/max_count)
    # print(structure)
    # print(crossed_pair_score, crossed_pair_quality_score)
    return [crossed_pair_score, crossed_pair_quality_score]

def calculateCorrelationCoefficient(structure, data, score_start_idx, score_end_idx, num_show, clip, method = "pearson"): 
    """ Calculates the correlation coefficient between a given structure and reactivity dataset
    
    data: [Ndesign x Nres x Ncond] Reactivity matrix. Assume last of Ncond has
    SHAPE data.
    structure:[Ndesign x Nres x Npackages] 0/1 map
        of paired/unpaired for each predicted structure
    good_idx = [list of integers] index of designs to use for correlation
        coefficients
    score_start_idx: index of data to start scoring from
    score_end_idx: index of data to end scoring at
    corr_type  = correlation coefficient type (default:'Pearson')
    num_show   = number of top algorithms to show (default: all)
    clip       = maximum value to clip at [Default no clipping]

    Outputs
        all_corr_coef = [Npackages] correlation coefficients for each package
        pkg_sort_idx = [Npackages] permutation of package indices that sort from best to
            worst by correlation coefficient
    """

#     if ~exist( 'corr_type','var') | length(corr_type)==0; corr_type = 'Pearson'; end;
#     if ~exist( 'num_show','var') | num_show == 0; num_show = length(structure_tags); end;
#     if ~exist( 'clip','var') | clip == 0; clip=Inf; end;
    
    if not isinstance(structure, str):
        print(f"Structure in dbn notation expected, got: {type(structure)}")
        return
    if not isinstance(data, list):
        print(f"List of reactivity values expected, got: {type(data)}")
        return
    if len(structure) != len(data):
        print(f"Structure and data array lengths don't match: {len(structure)} != {len(data)}")
        print(structure)
        return
    
    # Convert the input structure to a binary paired/unpaired list
    # We only have trustworthy data for the bases in the middle of the sequence
    # so we skip the blanked regions at the beginning and end
    prediction = [1 if char == "." else 0 for char in structure]
    try:
        data_df = pandas.Series(data[score_start_idx:score_end_idx+1])
        structure_df = pandas.Series(prediction[score_start_idx:score_end_idx+1])
        cc = data_df.corr(structure_df,method=method.lower())
    except:
        cc = float("nan")
        print(structure, data)
    # Use pandas built-in correlation function
    return cc

def calculateOpenKnotScore(row, prediction_tags):
    """ Calculates the OpenKnotScore metric for a structure and accompanying data

    The OpenKnotScore is a metric that estimates how likely a sequence is to contain 
    a pseudoknot structure. The score is derived by averaging the Eterna Classic Score
    (measure of structure match to reactivity data) and the Crossed Pair Quality Score
    (measure of reactivity support for crossed pairs) across an ensemble of several
    structure predictions from various predictive models.

    row: a pandas dataframe row containing the data for a single sequence including
      - DBN structures 
    prediction_tags: names of predictors (as present in row) to include in scoring
    scoring_region_length: length in nucleotides of the region that was scored
    """

    if len(prediction_tags) == 0: return

    ecs_tags = [f"{tag}_ECS" for tag in prediction_tags]

    # If there's no data, skip this row
    if not isinstance(row['reactivity'],list): return
    
    # Initialize a pandas series to hold the OKS scores
    df = pandas.Series(dtype='float64')

    # Generate a per-model OKS score
    for predictor in prediction_tags:
        # Grab the ECS score for this model, set OK to None if missing
        if pandas.isna(row[f"{predictor}_ECS"]):
            df[f"{predictor}_OKS"] = None
            print(f"ERROR: Missing ECS data for {predictor} in row {row.name}")
        ecs = row[f"{predictor}_ECS"]
        # Grab the CPQ score for this model, set OK to None if missing
        if isinstance(row[f"{predictor}_CPQ"],list):
            # The second value in the CPQ list is the CPQ score
            cpq = row[f"{predictor}_CPQ"][1]
            # Per-model OKS is the average of ECS and CPQ
            df[f"{predictor}_OKS"] = (0.5 * ecs) + (0.5 * cpq)
        else:
            df[f"{predictor}_OKS"] = None
            print(f"ERROR: Missing CPQ data for {predictor} in row {row.name}")

    # Generate per-sequence OKS by generating an ensemble of predicted structures
    # The OKS ensemble is all predictions with ECS scores within a given cutoff of 
    # the highest-scoring ECS. Essentially, all these predictions are close matches to 
    # the reactivity data.
    threshold = min(5, 5*100/(1 + row['score_end_idx'] - row['score_start_idx']))
    max_ecs = max(row[ecs_tags])
    # Get ensemble of structures with ECS scores within threshold of the highest-scoring ECS
    top_scoring_by_ecs = row[ecs_tags][row[ecs_tags] >= max_ecs - threshold].sort_values(ascending=False)
    # Get the CPQ scores for the ensemble structures
    top_scoring_index = [tag.replace("_ECS","") for tag in top_scoring_by_ecs.index.to_list()]
    top_scoring_cpq = row[[tag + "_CPQ" for tag in top_scoring_index]]

    # Calculate the ensemble OKS score
    avg_ecs = statistics.mean(top_scoring_by_ecs) 
    avg_cpq = statistics.mean([el[1] for el in top_scoring_cpq])
    df["ensemble_ECS"] = avg_ecs
    df["ensemble_CPQ"] = avg_cpq
    df["ensemble_OKS"] = 0.5*avg_ecs + 0.5*avg_cpq
    df["ensemble_tags"] = top_scoring_index
    df["ensemble_structures"] = list(row[top_scoring_index].values)
    df["ensemble_structures_ecs"] = list(top_scoring_by_ecs.values)
    
    return df