#!/usr/bin/bash

#SBATCH --job-name=
#SBATCH --output=./logs/Out%a.out
#SBATCH --error=./logs/Err%a.err
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --time=1-6:00:00
#SBATCH --partition=biochem,owners
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=

# If any errors are encountered, exit with a failure code
set -e
# Print commands to stderr as they're run
set -x

# Load standard libraries that our script will use
# See https://www.sherlock.stanford.edu/docs/software/modules/ for more details
module load python/3.12
source ../venv/bin/activate
# These libaries are needed for some predictors
module load gcc
module load glpk
module load mpfr

export RNA_ENV=$L_SCRATCH/rna-env
export ARNIEFILE=$RNA_ENV/arniefile.txt

# Setup local directories on the node's SCRATCH filesystem
mkdir -p $RNA_ENV
mkdir -p $RNA_ENV/tmp

# Only one process per node should handle setting up the environment
# to avoid doing unnecessary work (or stepping on each other) so we'll
# use a file lock
(
    if flock -n 200; then
        # Only do this if we haven't already successfully done the transfer on this node
        if [ ! -f $RNA_ENV/setup.done ]; then
            # Copy our rna environment to the node's local directories we just created
            # This helps us avoid issues we've run into with large job arrays all trying to access
            # GROUP_HOME simultaneously, and slowing down Sherlock access for other group members
            cp -r $SCRATCH/rna-env/predictors $RNA_ENV
            # Write package locations to our arniefile
            cat > $ARNIEFILE <<- EOM
TMP: $RNA_ENV/tmp
contrafold_2: $RNA_ENV/predictors/PK/eternagame-EternaFold-d65ecd41aec23594ab10ddf4f49eb16066405a35/src
eternafold: $RNA_ENV/predictors/PK/eternagame-EternaFold-d65ecd41aec23594ab10ddf4f49eb16066405a35/src
e2efold: $RNA_ENV/predictors/PK/e2efold/e2efold_productive
e2efold_conda_env: $RNA_ENV/predictors/miniconda3/envs/e2efold/bin
hotknots: $RNA_ENV/predictors/PK/HotKnots_v2.0/bin
ipknot: $RNA_ENV/predictors/PK/ipknot/build
knotty: $RNA_ENV/predictors/PK/Knotty
pknots: $RNA_ENV/predictors/PK/PKNOTS/bin
spotrna: $RNA_ENV/predictors/PK/SPOT-RNA
spotrna_conda_env: $RNA_ENV/predictors/miniconda3/envs/spotrna/bin
rnastructure: $RNA_ENV/predictors/RNAstructure/exe
vienna_2: $RNA_ENV/predictors/ViennaRNA-2.4.16/src/bin
EOM
            touch $RNA_ENV/setup.done
        fi
    else
        # Wait until the process actively doing the transfer is done
        flock --timeout $((60*30)) 200
        # Make sure the copy process actually finished
        if [ ! -f $RNA_ENV/setup.done ]; then
            echo "rna-env setup was left in an inconsistent state by the job on this node that was responsible for setting it up. Please resubmit this job"
            exit 1
        fi
    fi
) 200>$RNA_ENV/setup.lock

# Not currently managed by arnie
export CONTRAFOLD_2_PARAMS_PATH=$RNA_ENV/predictors/PK/contrafold-se/src/contrafold.params.complementary
export HFOLD_PATH=$RNA_ENV/predictors/Shapify
# For rnastucture
export DATAPATH=$RNA_ENV/predictors/RNAstructure/data_tables
# Nupack requires this to be set in order to function correctly
# We skip it in the arniefile since arnie will prefer the env var over the arniefile anyways
export NUPACKHOME=$RNA_ENV/predictors/PK/nupack3.0.6

python3 -u ./GeneratePredictions.py $SLURM_ARRAY_TASK_ID $SLURM_JOB_NAME
