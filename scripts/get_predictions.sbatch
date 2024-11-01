#!/usr/bin/bash

#SBATCH --job-name=
#SBATCH --output=./logs/Out%a.out
#SBATCH --error=./logs/Err%a.err
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=
#SBATCH --time=1440:00
#SBATCH --partition=biochem,owners
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=

# Load standard libraries that our script will use
# See https://www.sherlock.stanford.edu/docs/software/modules/ for more details
module load python/3.6.1
module load py-numpy/1.18.1_py36 
module load py-pandas/1.0.3_py36
module load py-scipy/1.4.1_py36
# These libaries are needed for some predictors
module load gcc
module load glpk
module load mpfr
# Load your bash setup; particularly your arnie environment variables
source ~/.bashrc

# Setup local directories on the node's SCRATCH filesystem
mkdir $L_SCRATCH/rna-env
mkdir $L_SCRATCH/rna-env/tmp
mkdir $L_SCRATCH/rna-env/predictors

# Copy our rna environment to the node's local directories we just created
# This helps us avoid issues we've run into with large job arrays all trying to access
# GROUP_HOME simultaneously, and slowing down Sherlock access for other group members
cp -r $SCRATCH/rna-env/arnie $L_SCRATCH/rna-env/arnie
cp -r $SCRATCH/rna-env/predictors/ $L_SCRATCH/rna-env/predictors/

# Update environment variables to point arnie to the local filesystem
export ARNIE_TMP=$L_SCRATCH/rna-env/tmp

export contrafold_2_PATH=$L_SCRATCH/rna-env/predictors/PK/eternagame-EternaFold-d65ecd41aec23594ab10ddf4f49eb16066405a35/src
export eternafold_PATH=$L_SCRATCH/rna-env/predictors/PK/eternagame-EternaFold-d65ecd41aec23594ab10ddf4f49eb16066405a35/src
# For e2efold
export e2efold_PATH=$L_SCRATCH/rna-env/predictors/PK/e2efold/e2efold_productive
export e2efold_conda_env=$L_SCRATCH/rna-env/predictors/miniconda3/envs/e2efold/bin
export hotknots_PATH=$L_SCRATCH/rna-env/predictors/PK/HotKnots_v2.0/bin
export ipknot_PATH=$L_SCRATCH/rna-env/predictors/PK/ipknot/build
export knotty_PATH=$L_SCRATCH/rna-env/predictors/PK/Knotty
export pknots_PATH=$L_SCRATCH/rna-env/predictors/PK/PKNOTS/bin
export spotrna_PATH=$L_SCRATCH/rna-env/predictors/PK/SPOT-RNA
export spotrna_conda_env=$L_SCRATCH/rna-env/predictors/miniconda3/envs/spotrna/bin
# For NuPACK
export NUPACKHOME=$L_SCRATCH/rna-env/predictors/PK/nupack3.0.6
# For rnastucture
export rnastructure_PATH=$L_SCRATCH/rna-env/predictors/RNAstructure/exe
export DATAPATH=$L_SCRATCH/rna-env/predictors/RNAstructure/data_tables
export vienna_2_PATH=$L_SCRATCH/rna-env/predictors/M2SEQ/ViennaRNA-2.4.16/src/bin

python3 ./GeneratePredictions.py $SLURM_ARRAY_TASK_ID $SLURM_JOB_NAME
