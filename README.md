# OpenKnotScorePipeline
Python pipeline to generate OpenKnotScores for Eterna sequence libraries

## How to use this pipeline
The notebooks in the notebooks directory are ordered and meant to be run to transform the data as it needs to be modified at various stages in the pipeline. Generally,
1. We start with an RDAT file containing reactivity data for an RNA sequence library. We [extract the library](/notebooks/1.RDATtoDataFrame.ipynb) (sequence, reactivity data, reads, etc) into a dataframe. If you're planning to process the dataset in batch mode on Sherlock, refer to [this notebook](/notebooks/1b.SplitDataForSherlockProcessing.ipynb) to split the dataframe into multiple subsets.
2. Next, we [compute silico predictions](/notebooks/2.AddSilicoPredictions.ipynb) using a range of RNA structure predction algorithms. The [actual script](/scripts/get_predictions.sbatch) to generate these predictions is available; [this notebook](/notebooks/2.AddSilicoPredictions.ipynb) provides more details if you're planning to run on Sherlock. If you do use batch processing on Sherlock to generate the predictions, you'll need to [collate the processed subset files](/notebooks/2b.CollateSubsets.ipynb) into a single dataframe for the next step.
3. Now that the sequence library has structure predictions, we can [calculate the OpenKnotScore](/notebooks/3.CalculateOpenKnotScore.ipynb) for each sequence. This step creates a new dataframe with a bunch of scoring details added to the sequence library.
4. Finally, we [extract relevant scoring details](/notebooks/4.DataFrametoRDAT.ipynb) from the library and add them to the original RDAT file for upload to Eterna.

## Notes on Sherlock processing
If you plan on running these notebooks/scripts on Stanford's Sherlock computing cluster (which is a good idea if you have a large sequence library to process), you may also want to review https://daslab.github.io/arnie/#/sherlock/environment for some tips on how to properly set up an `arnie` environment on Sherlock. The structure generation relies on having a wide range of folding algorithms available, and Python environments on Sherlock can be tricky.