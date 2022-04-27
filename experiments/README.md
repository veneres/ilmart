# ILMART experiments
In this folder are stored the main scripts to replicate the results described in the related paper.

## Structure of the folder
For each model analyzed we created a folder containing the scripts to trains and evaluate the models.
That is, we have 4 different folders:
- `ebm` contains the scripts for EBM
- `ilmart` contains the scripts for our proposed method
- `lmart` contains the script for the full LambdaMART
- `nrgam` contains the script NeuralRankGAM

In addition, for `ilmart`, a notebook is shared to replicate part of the analysis made on the model.

We also provide the pretrained models in a 
[Google Drive folder](https://drive.google.com/file/d/1fjk1qtS8G6aMP9xxPfBSCo9LvZB5xFh0/view?usp=sharing)
that is possible to download and extract automatically and in the correct position with the Python script `download_files.py`.
`download_files.py` also download the NDCG results of the evaluation of each model over the tests sets, 
and it is used inside the notebook `stat_analysis.ipynb` to perform the statistical significance analysis.

Last but not least, we provide the exported conda environment in `environment.yml` and a sample docker file
`Dockerfile` that simulate the settings used in our experiments.

## Disclaimers
1) The `Dockerfile` has been provided only to give the idea on how the machine has been set up, the pretrained
models have not been generated using that particular `Dockerfile`.

2) Don't be surprised if you get lower results on Yahoo using EBM, we have identified some reproducibility issues
that are probably related to a bug in handling the categorical features 
(possibly related to this [issue](https://github.com/interpretml/interpret/issues/318)).
We are still investigating on the issue, however, the discrepancies do not change the core idea of the paper and
it is not related with the efficacy of `ilmart`.





