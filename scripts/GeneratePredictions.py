import os, sys, time
import pandas
import numpy

# Import various prediction functions from arnie
from arnie.bpps import bpps
from arnie.mfe import mfe
from arnie.pk_predictors import pk_predict
from arnie.pk_predictors import pk_predict_from_bpp

from arnie.utils import convert_dotbracket_to_bp_list, convert_bp_list_to_dotbracket

##############################
# SETUP
##############################

# Define which prediction algorithms you want to use
predictors = {
  "vienna2": True, 
  "contrafold2": True, 
  "eternafold": True, 
  "rnastructure_mfe": True, 
  "e2efold": False, # Works, but poor performance
  "hotknots": True, 
  "ipknots": True, 
  "knotty": True, 
  "pknots": True, 
  "spotrna": True, 
  "spotrna2": False, # Fails
  "shapify-hfold": True,
  "nupack_pk": True,
  "rnastructure+SHAPE": True, # Requires reactivity data
  "shapeknots": True, # Requires reactivity data
}
heuristic_predictions = {
  'vienna_2': True,
  'eternafold': True,
  'contrafold_2': True,
}

# Define path variables
# We use the job name provided in the SLURM submission script, so make sure the job name you enter
# is the same as the job name used when you generated the subset data files
job = int(sys.argv[2])
dataDir = f'{os.environ["SCRATCH"]}/{job}/data'

# The Sherlock batch array script will pass in a number indicating which job in the array this is
count = int(sys.argv[1])-1
# We use that job array number to pick which subset file this job processes
df = pandas.read_pickle(f'{dataDir}/{count:03}.pkl')

# Clean extra whitespace from sequence
df["sequence"] = df["sequence"].apply(lambda x : x.strip())

# If the data is read in from a csv, we need to convert reactivity data from a string to a list of floats
if type(df["reactivity"].iloc[0]) == str:
  df["reactivity"] = df["reactivity"].apply(lambda x : numpy.fromstring(x[1:-1], sep=','))

# Define BLANK_OUT5 and BLANK_OUT3 variables
BLANK_OUT5 = 26
BLANK_OUT3 = len(df["sequence"].iloc[0]) - BLANK_OUT5 - len(df["reactivity"].iloc[0])

# For HFold; unnecessary once HFold is moved to arnie
import re
import subprocess as sp
from subprocess import PIPE

# TODO: Move into arnie_file
hfold_location = "/home/groups/rhiju/thomask/FoldingAlgorithms/Shapify"

##############################
# RUN PREDICTORS
##############################

# Start processing predictors; we provide rough time estimates along with each structure prediction
# because it helps for estimating prediction time for the whole dataset
total_start = time.perf_counter()

def process_vienna2(seq):
  # MFE Structure - Vienna2
  start = time.perf_counter()
  try:
    structure = mfe(seq, package="vienna_2", pseudoknots=True)
  except Exception as e: 
    print(e)
    structure = 'x'
  end = time.perf_counter()
  return pandas.Series([ structure, end-start ], index=['vienna2_PRED', 'vienna2_time'])
if (predictors.get("vienna2") and "vienna2_PRED" not in df.columns):
  print("Processing Vienna MFE")
  df = df.join(df["sequence"].apply(process_vienna2))
  # Checkpoint save
  df.to_pickle(f'{dataDir}/{count:03}.pkl')

def process_contrafold2(seq):
  # MFE Structure - Contrafold2
  start = time.perf_counter()
  try:
    structure = mfe(seq, package="contrafold_2", pseudoknots=True, param_file="$SCRATCH/rna-env/predictors/PK/contrafold-se/src/contrafold.params.complementary")
  except Exception as e: 
    print(e)
    structure = 'x'
  end = time.perf_counter()
  return pandas.Series([ structure, end-start], index=['contrafold2_PRED', 'contrafold2_time'])
if (predictors.get("contrafold2") and "contrafold2_PRED" not in df.columns):
  print("Processing Contrafold MFE")
  df = df.join(df["sequence"].apply(process_contrafold2))
  # Checkpoint save
  df.to_pickle(f'{dataDir}/{count:03}.pkl')

def process_eternafold(seq):
  # MFE Structure - Eternafold
  start = time.perf_counter()
  try:
    structure = mfe(seq, package="eternafold", pseudoknots=True)
  except Exception as e:
    print(e)
    structure = 'x'
  end = time.perf_counter()
  return pandas.Series([ structure, end-start], index=['eternafold_PRED', 'eternafold_time'])
if (predictors.get("eternafold") and "eternafold_PRED" not in df.columns):
  print("Processing Eternafold MFE")
  df = df.join(df["sequence"].apply(process_eternafold))
  # Checkpoint save
  df.to_pickle(f'{dataDir}/{count:03}.pkl')

def process_rnastructure(seq):
  # MFE Structure - Eternafold
  start = time.perf_counter()
  try:
    structure = mfe(seq, package="rnastructure")
  except Exception as e:
    print(e)
    structure = 'x'
  end = time.perf_counter()
  return pandas.Series([ structure, end-start], index=['rnastructure_PRED', 'rnastructure_time'])
if (predictors.get("rnastructure_mfe") and "rnastructure_PRED" not in df.columns):
  print("Processing RNAStructure")
  df = df.join(df["sequence"].apply(process_rnastructure))
  # Checkpoint Save
  df.to_pickle(f'{dataDir}/{count:03}.pkl')

def process_e2efold(seq):
  # PK Structure - e2efold
  start = time.perf_counter()
  try:
    structure = pk_predict(seq,"e2efold")
  except Exception as e:
    print(e)
    structure = 'x'
  end = time.perf_counter()
  return pandas.Series([ structure, end-start], index=['e2efold_PRED', 'e2efold_time'])
if (predictors.get("e2efold") and "e2efold_PRED" not in df.columns):
  print("Processing e2efold")
  df = df.join(df["sequence"].apply(process_e2efold))
  # Checkpoint Save
  df.to_pickle(f'{dataDir}/{count:03}.pkl')

def process_hotknots(seq):
  # PK Structure - hotknots
  start = time.perf_counter()
  try:
    structure = pk_predict(seq,"hotknots")
  except Exception as e:
    print(e)
    structure = 'x'
  end = time.perf_counter()
  return pandas.Series([ structure, end-start ], index=['hotknots_PRED', 'hotknots_time'])  
if (predictors.get("hotknots") and "hotknots_PRED" not in df.columns):
  print("Processing Hotknots")
  df = df.join(df["sequence"].apply(process_hotknots))
  # Checkpoint save
  df.to_pickle(f'{dataDir}/{count:03}.pkl')

def process_ipknots(seq):
  # PK Structure - ipknot
  start = time.perf_counter()
  try:
    structure = pk_predict(seq,"ipknot")
  except Exception as e:
    print(e)
    structure = 'x'
  end = time.perf_counter()
  return pandas.Series([ structure, end-start], index=['ipknots_PRED', 'ipknots_time'])
if (predictors.get("ipknots") and "ipknots_PRED" not in df.columns):
  print("Processing IPKnots")
  df = df.join(df["sequence"].apply(process_ipknots))
  # Checkpoint save
  df.to_pickle(f'{dataDir}/{count:03}.pkl')
    
def process_knotty(seq):
  # PK Structure - knotty
  start = time.perf_counter()
  try:
    structure = pk_predict(seq,"knotty")
  except Exception as e:
    print(e)
    structure = 'x'
  end = time.perf_counter()
  return pandas.Series([ structure, end-start], index=['knotty_PRED', 'knotty_time'])
if (predictors.get("knotty") and "knotty_PRED" not in df.columns):
  print("Processing Knotty")
  df = df.join(df["sequence"].apply(process_knotty))
  # Checkpoint Save
  df.to_pickle(f'{dataDir}/{count:03}.pkl')
    
def process_pknots(seq):
  # PK Structure - pknots
  start = time.perf_counter()
  try:
    structure = pk_predict(seq,"pknots")
  except Exception as e:
    print(e)
    structure = 'x'
  end = time.perf_counter()
  return pandas.Series([ structure, end-start], index=['pknots_PRED', 'pknots_time'])
if (predictors.get("pknots") and "pknots_PRED" not in df.columns):
  print("Processing PKnots")
  df = df.join(df["sequence"].apply(process_pknots))
  # Checkpoint save
  df.to_pickle(f'{dataDir}/{count:03}.pkl')
    
def process_spotrna(seq):
  # PK Structure - spotrna
  start = time.perf_counter()
  try:
    structure = pk_predict(seq,"spotrna")
  except Exception as e:
    print(e)
    structure = 'x'
  end = time.perf_counter()
  return pandas.Series([ structure, end-start], index=['spotrna_PRED', 'spotrna_time'])
if (predictors.get("spotrna") and "spotrna_PRED" not in df.columns):
  print("Processing SpotRNA")
  df = df.join(df["sequence"].apply(process_spotrna))
  # Checkpoint save
  df.to_pickle(f'{dataDir}/{count:03}.pkl')
    
def process_spotrna2(seq):
  # PK Structure - spotrna
  start = time.perf_counter()
  try:
    structure = pk_predict(seq,"spotrna2")
  except Exception as e:
    print(e)
    structure = 'x'
  end = time.perf_counter()
  return pandas.Series([ structure, end-start], index=['spotrna2_PRED', 'spotrna2_time'])
if (predictors.get("spotrna2") and "spotrna2_PRED" not in df.columns):
  print("Processing SpotRNA2")
  df = df.join(df["sequence"].apply(process_spotrna2))
  # Checkpoint save
  df.to_pickle(f'{dataDir}/{count:03}.pkl')
    
def process_nupack_pk(seq):
  # PK Structure - Nupack PK Predictor
  start = time.perf_counter()
  try:
    structure = pk_predict(seq,"nupack")
  except Exception as e:
    print(e)
    structure = 'x'
  end = time.perf_counter()
  return pandas.Series([ structure, end-start], index=['nupack_pk_PRED', 'nupack_pk_time'])
if (predictors.get("nupack_pk") and "nupack_pk_PRED" not in df.columns):
  print("Processing NuPACK-PK")
  df = df.join(df["sequence"].apply(process_nupack_pk))
  # Checkpoint save
  df.to_pickle(f'{dataDir}/{count:03}.pkl')
    
def process_heuristic(seq, package):
  print(f"Processing {package}_heuristic")
  t0 = time.perf_counter()
  if (package == "nupack-pk"):
    bpp = bpps(seq, package='nupack', pseudo=True)
  else:
    bpp = bpps(seq, package=package)
  t1 = time.perf_counter()
  structure_t = pk_predict_from_bpp(bpp,heuristic="threshknot")
  t2 = time.perf_counter()
  structure_h = pk_predict_from_bpp(bpp,heuristic="hungarian")
  t3 = time.perf_counter()
  return pandas.Series([structure_t, t2-t0, structure_h, (t1-t0)+(t3-t2)], index=[f"{package}.TK_PRED", f"{package}.TK_time", f"{package}.HN_PRED", f"{package}.HN_time"])

for package in heuristic_predictions.keys():
  if (heuristic_predictions[package] and f"{package}.TK_PRED" not in df.columns):
    df = df.join(df["sequence"].apply(process_heuristic, args=(package,)))
    # Checkpoint save
    df.to_pickle(f'{dataDir}/{count:03}.pkl')
        
# TODO: Move Shapify into PK_Predictors
def process_shapify(seq):
  start = time.perf_counter()
  try:
    hfold_return = sp.run([hfold_location+"/HFold_iterative", "--s", seq.strip()], stdout=PIPE, stderr=PIPE, encoding="utf-8")

    # Pull structure out of Shapify output
    match = re.search(r"Result_0: (.*) \n", hfold_return.stdout)
    structure = match.group(1) if match else "ERR"
    
    # Sanitize dbn structure to match our conventions 
    structure = convert_bp_list_to_dotbracket(convert_dotbracket_to_bp_list(structure, allow_pseudoknots=True), len(structure))
  except Exception as e:
    print(e)
    structure = 'x'
  end = time.perf_counter()
  return pandas.Series([ structure, end-start], index=['shapify-hfold_PRED', 'shapify-hfold_time'])
if (predictors.get("shapify-hfold") and "shapify-hfold_PRED" not in df.columns):
  print("Processing Shapify-HFold")
  df = df.join(df["sequence"].apply(process_shapify))
  # Checkpoint save
  df.to_pickle(f'{dataDir}/{count:03}.pkl')
    
# Shapeknots is a little different from the other predictors; because it uses the sequence and the reactivity data    
# we provide the entire row, rather than just the sequence, to the processing function
def process_SHAPEKNOTS(row):
  # Predicted Structure - ShapeKnots
  start = time.perf_counter()
  try:
    seq = row["sequence"]
    # Set up the reactivity array that will be passed to shapeknots
    # arnie expects a list of float values; we need to account for the leader region with invalid data,
    # so we don't provide the reactivity data directly stored in the row
    reactivities = [0] * len(seq)
    for index, value in enumerate(row["reactivity"]):
      reactivities[index+BLANK_OUT5] = value

    structure = mfe(seq, package="rnastructure", shape_signal=reactivities, pseudo=True)
  except Exception as e:
    print(e)
    structure = 'x'
  end = time.perf_counter()
  return pandas.Series([ structure, end-start], index=['shapeknots_PRED', 'shapeknots_time'])
if (predictors.get("shapeknots") and "shapeknots_PRED" not in df.columns):
  print("Processing ShapeKnots")
  df = df.join(df.apply(process_SHAPEKNOTS,axis=1))

def process_rnastructure_SHAPE(row):
  # Predicted Structure - RNAstructure + SHAPE data
  start = time.perf_counter()
  try:
    seq = row["sequence"]
    # Set up the reactivity array that will be passed to rnastructure
    # arnie expects a list of float values; we need to account for the leader region with invalid data,
    # so we don't provide the reactivity data directly stored in the row
    reactivities = [0] * len(seq)
    for index, value in enumerate(row["reactivity"]):
      reactivities[index+BLANK_OUT5] = value

    structure = mfe(seq, package="rnastructure", shape_signal=reactivities)
  except Exception as e:
    print(e)
    structure = 'x'
  end = time.perf_counter()
  return pandas.Series([ structure, end-start], index=['rnastructure+SHAPE_PRED', 'rnastructure+SHAPE_time'])
if (predictors.get("rnastructure+SHAPE") and "rnastructure+SHAPE_PRED" not in df.columns):
  print("Processing RNAstructure + SHAPE data")
  df = df.join(df.apply(process_rnastructure_SHAPE,axis=1))

# Final save with all the predictions
df.to_pickle(f'{dataDir}/{count:03}.pkl')

# Print total processing time for future estimates
total_end = time.perf_counter()
print(total_end - total_start)