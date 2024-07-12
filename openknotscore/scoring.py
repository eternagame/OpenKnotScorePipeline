import pandas

def calculateEternaClassicScore(structure, data, BLANK_OUT5, BLANK_OUT3):
    """Calculates an Eterna score for a predicted structure and accompanying reactivity dataset
    
    data: a list of reactivity values, normalized from 0 to 1 (~90th percentile)
    structure: a list of predicted values in dot bracket notation
    BLANK_OUT5: gray out this number of 5' residues
    BLANK_OUT3: gray out this nymber of 3' residues
    """
    
    if not isinstance(structure, str):
        print(f"Structure in dbn notation expected, got: {type(structure)}")
        return
    if not isinstance(data, list):
        print(f"List of reactivity values expected, got: {type(data)}")
        print(data)
        return
    if len(structure) != len(data) + BLANK_OUT5 + BLANK_OUT3:
        print(f"Structure and data array lengths don't match: {len(structure)} != {len(data)} + {BLANK_OUT5} + {BLANK_OUT3}")
        print(structure)
        return
#     # If the structure being passed in is not a string, exit
#     assert isinstance(structure, str), f"Structure in dbn notation expected, got: {type(structure)}"
#     # Check that the structure length matches the length of data values + flanking regions
#     assert len(structure) == len(data) + BLANK_OUT5 + BLANK_OUT3 , f"Structure and data array lengths don't match: {len(structure)} != {len(data)} + {BLANK_OUT5} + {BLANK_OUT3}"
    
    # Cutoff values; minimum value threshold of paired or unpaired in SHAPE data
    threshold_SHAPE_fixed = 0.5
    min_SHAPE_fixed = 0.0
    
    # Convert the input structure to a binary paired/unpaired list
    # We only have trustworthy data for the bases in the middle of the sequence
    # so we skip the blanked regions at the beginning and end
    prediction = [1 if char == "." else 0 for char in structure][BLANK_OUT5:(len(structure) - BLANK_OUT3)]

    correct_hit = [0] * len(data)
    
    # Loop over each base we are testing for correlation 
    for i in range(len(data)):
        # If the prediction is the base is unpaired
        if (prediction[i] == 1):
        # We check that the data passes a cutoff, and add it to the correct_hit list if true
            if (data[i] > (0.25*threshold_SHAPE_fixed + 0.75*min_SHAPE_fixed)):
                correct_hit[i] = 1
        else:
        # The prediction is the base is paired, we check that the data is below the threshold value
            if (data[i] < threshold_SHAPE_fixed):
                correct_hit[i] = 1
    
    # To get the eterna score, we sum the correct_hit list to figure out how many predicted bases
    # matched the data, then divide by the total number of scored bases and normalize to 100
    eterna_score = ( sum(correct_hit) / len(data) ) * 100
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
        if (bp[0] < bp[1]):
            # Now we loop over the list again, grabbing base pairs to compare
            for next_bp in bps:
                # This checks whether the original bp and the compared bp cross
                # If they do, we add base positions to the crossed residue list
                if (bp[0] < next_bp[0] & next_bp[0] < bp[1] & bp[1] < next_bp[1]) | ( next_bp[0] < bp[0] & bp[0] < next_bp[1] & next_bp[1] < bp[1]):
                    crossed_res.append(bp[0])
                    crossed_res.append(bp[1])
                    crossed_res.append(next_bp[0])
                    crossed_res.append(next_bp[1])
                    
    # Returns a unique list, since crossed residues may contain duplicate values
    return list(set(crossed_res)) if len(crossed_res) else []

def remove_bps_in_blanked_regions(bps, num_res, BLANK_OUT5, BLANK_OUT3):
    filtered_bps = []
    for bp in bps:
        if (bp[0] < bp[1]):
            if (bp[0] <= BLANK_OUT5): continue
            if (bp[1] <= BLANK_OUT5): continue
            if (bp[0] > num_res - BLANK_OUT3): continue
            if (bp[1] > num_res - BLANK_OUT3): continue
            filtered_bps.append(bp)
    return filtered_bps

def calculateCrossedPairQualityScore(structure, data, BLANK_OUT5, BLANK_OUT3):
    """Calculates the crossed pair quality score for a DBN structure and accompanying dataset
    
    crossed_pair_score = 
    100 * (number of residues in crossed pairs with data < 0.25) /
        [0.7*(length of region with data - 20)]

    crossed_pair_quality_score =
    100 * (number of residues in crossed pairs with data < 0.25) /
        ( number of residues modeled to be crossed pairs in structure )
           
    data: [Nres] data, normalized to go from 0 to 1 (~90th percentile)
    structure: string in dot-bracket notation for structure, with pseudoknots
    BLANK_OUT5: gray out this number of 5' residues
    BLANK_OUT3: gray out this number of 3' residues
    """
    if not isinstance(structure, str):
        print(f"Structure in dbn notation expected, got: {type(structure)}")
        return
    if not isinstance(data, list):
        print(f"List of reactivity values expected, got: {type(data)}")
        return
    if len(structure) != len(data) + BLANK_OUT5 + BLANK_OUT3:
        print(f"Structure and data array lengths don't match: {len(structure)} != {len(data)} + {BLANK_OUT5} + {BLANK_OUT3}")
        return

    padded_data = [float('nan')]*BLANK_OUT5 + data + [float('nan')]*BLANK_OUT3
    
    threshold_SHAPE_fixed_pair = 0.25
    
    crossed_pair_score = 0
    crossed_pair_quality_score = 0
    
    # Some algorithms will produce a string of all x's when they fail on a sequence
    # This checks for a structure string that is all x characters and returns if true
    failed_structure = [char == "x" for char in structure]
    if (all(failed_structure)):
            return [crossed_pair_score, crossed_pair_quality_score]

    # Remove singlet base pairs
    stems = get_helices(structure)
    for stem in stems:
        if (len(stem) == 1):
            stems.remove(stem)
    # Convert the list of helices into a list of base pairs (flatten 3D list to 2D list)
    bp_list = [item for sublist in stems for item in sublist]
    
    # Get indexes for bases in crossed pairs
    crossed_res = identify_crossing_bps(bp_list)

    # Filter out base pairs that involve residues in the blanked out flanking regions
    bps_filtered = remove_bps_in_blanked_regions(bp_list, len(structure), BLANK_OUT5, BLANK_OUT3)
    crossed_res_filtered = identify_crossing_bps(bps_filtered)
    
    num_crossed_pairs  = 0
    crossed_pair_quality_score = 0 
    total_cross_res = 0 # Unused?
    max_count = 0
    
    for i in crossed_res:
        # Skip if the base index is in a flanking region
        if i < BLANK_OUT5: continue
        if i > len(structure) - BLANK_OUT3: continue
        
        max_count = max_count + 1
        if ( padded_data[i] < threshold_SHAPE_fixed_pair):
            if i in crossed_res_filtered:
                num_crossed_pairs = num_crossed_pairs + 1
            else:
                num_crossed_pairs = num_crossed_pairs + 0.5
                
    data_region_length = len(data)
    # TODO: Explain heuristic
    max_crossed_pairs = 0.7 * max(data_region_length - 20, 20)
    crossed_pair_score = 100 * min( num_crossed_pairs/max_crossed_pairs, 1.0)

    if max_count > 0:
        crossed_pair_quality_score = 100 * (num_crossed_pairs/max_count)
    # print(structure)
    # print(crossed_pair_score, crossed_pair_quality_score)
    return [crossed_pair_score, crossed_pair_quality_score]

def calculateCorrelationCoefficient(structure, data, BLANK_OUT5, BLANK_OUT3, num_show, clip, method = "pearson"): 
    """ Calculates the correlation coefficient between a given structure and reactivity dataset
    
    data: [Ndesign x Nres x Ncond] Reactivity matrix. Assume last of Ncond has
    SHAPE data.
    structure:[Ndesign x Nres x Npackages] 0/1 map
        of paired/unpaired for each predicted structure
    good_idx = [list of integers] index of designs to use for correlation
        coefficients
    BLANK_OUT5 = gray out this number of 5' residues
    BLANK_OUT3 = gray out this number of 3' residues 
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
    if len(structure) != len(data) + BLANK_OUT5 + BLANK_OUT3:
        print(f"Structure and data array lengths don't match: {len(structure)} != {len(data)} + {BLANK_OUT5} + {BLANK_OUT3}")
        return
    
    # Convert the input structure to a binary paired/unpaired list
    # We only have trustworthy data for the bases in the middle of the sequence
    # so we skip the blanked regions at the beginning and end
    prediction = [1 if char == "." else 0 for char in structure][BLANK_OUT5:(len(structure) - BLANK_OUT3)]
    try:
        data_df = pandas.Series(data)
        structure_df = pandas.Series(prediction)
        cc = data_df.corr(structure_df,method=method.lower())
    except:
        cc = float("nan")
        print(structure, data)
    # Use pandas built-in correlation function
    return cc

import statistics

def calculateOpenKnotScore(row, prediction_tags):
    """ Calculates the OpenKnotScore metric for a structure and accompanying data

    row: a pandas dataframe row containing the data for a single sequence including
      - DBN structures 
    
    data: [Ndesign x Nres x Ncond] Reactivity matrix. Assume last of Ncond has
    SHAPE data.
    structure:[Ndesign x Nres x Npackages] 0/1 map
        of paired/unpaired for each predicted structure
    good_idx = [list of integers] index of designs to use for correlation
        coefficients
    BLANK_OUT5 = gray out this number of 5' residues
    BLANK_OUT3 = gray out this number of 3' residues 
    corr_type  = correlation coefficient type (default:'Pearson')
    num_show   = number of top algorithms to show (default: all)
    clip       = maximum value to clip at [Default no clipping]

    Outputs
        all_corr_coef = [Npackages] correlation coefficients for each package
        pkg_sort_idx = [Npackages] permutation of package indices that sort from best to
            worst by correlation coefficient
    """

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
            print(f"ERROR: Missing ECS data for {predictor} in row {row['id']}")
        ecs = row[f"{predictor}_ECS"]
        # Grab the CPQ score for this model, set OK to None if missing
        if isinstance(row[f"{predictor}_CPQ"],list):
            # The second value in the CPQ list is the CPQ score
            cpq = row[f"{predictor}_CPQ"][1]
            # Per-model OKS is the average of ECS and CPQ
            df[f"{predictor}_OKS"] = (0.5 * ecs) + (0.5 * cpq)
        else:
            df[f"{predictor}_OKS"] = None
            print(f"ERROR: Missing CPQ data for {predictor} in row {row['id']}")

    # Generate per-sequence OKS by generating an ensemble of predicted structures
    # The OKS ensemble is all predictions with ECS scores within a given cutoff of 
    # the highest-scoring ECS. Essentially, all these predictions are close matches to 
    # the reactivity data.
    threshold = 2.5;
    max_ecs = max(row[ecs_tags])
    # Get ensemble of structures with ECS scores within threshold of the highest-scoring ECS
    top_scoring_by_ecs = row[ecs_tags][row[ecs_tags] >= max_ecs - threshold].sort_values(ascending=False, inplace=True)
    # Get the CPQ scores for the ensemble structures
    top_scoring_index = [tag.replace("_ECS","_CPQ") for tag in top_scoring_by_ecs.index.to_list()]
    top_scoring_cpq = row[top_scoring_index]

    # Calculate the ensemble OKS score
    avg_ecs = statistics.mean(top_scoring_by_ecs) 
    avg_cpq = statistics.mean([el[1] for el in top_scoring_cpq])
    df["ensemble_ECS"] = avg_ecs
    df["ensemble_CPQ"] = avg_cpq
    df["ensemble_OKS"] = 0.5*avg_ecs + 0.5*avg_cpq
    return df