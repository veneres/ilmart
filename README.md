# ILMART: Interpretable LambdaMART
This repository contains the proof of concept of ILMART, a modified version of LambdaMART to create an interpretable
learning-to-rank model.
Major details about the algorithms can be found in the related paper, see the citation section.

## Version history
- 1.0.0: First version of the framework, available at https://github.com/veneres/ilmart/releases/tag/v1.0.0 and related to the SIGIR 2022 short paper [ILMART: Interpretable Ranking with Constrained LambdaMART](https://dl.acm.org/doi/10.1145/3477495.3531840).
- 2.0.0: Second improved version, currently under peer-review.

## Installation
There are two main steps to install the packages
1) Install a modified version of [LightGBM](https://github.com/veneres/LightGBM)
2) Install the sample package of ILMART

### Install the modified version of LightGBM:
The algorithm works with a modified version of [LightGBM](https://github.com/veneres/LightGBM). You can download it
as a submodule of this GIT repository cloning using:

```bash
git clone --recurse-submodules -b ilmart https://github.com/veneres/LightGBM.git 
```
Then, switch to the folder containing the installer of the python packaged and run this command to install LightGBM (Linux and MacOS):

```bash
sh ./build-python.sh install
```

### Install ILMART:
After having installed the modified version of LightGBM, you can install and test ILMART simply using:

```bash
pip install -e .
```

## Usage
ILMART is only a proof of concept and has been developed as a simple wrapper of LightGBM. 
To fit a model with ILMART you can simply run:

```python
ilmart = Ilmart()
ilmart.fit(lgbm_params, num_boosting_rounds, train, vali)
```
where `lgbm_params` are the standard LightGBM 
[training parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html), `num_boosting_rounds` are the 
boosting rounds used to train the `main effects` and the `interaction effects` each, and `train` and `vali` are two
instances of [rankeval datasets](https://github.com/hpclab/rankeval) used during the training.

Finally, you can get the underline booster using:

```bash
ilmart.get_model()
```



## Reproducibility
Some simple scripts to replicate the major results presented in the related paper are available inside the 
[experiments folder](experiments).
In all the subfolders, you can find actual command to replicate the results described in the related paper run in a 
docker container created with the Dockerfile that you can find in the main folder of this repository.
You can run the docker container using:

```bash
# Build the docker container
docker build --build-arg UID=$(id -u) --build-arg UNAME=$(id -un) -t ilmart:v2.0 .
# Run the docker container
docker run \
      --user $(id -un) \
      --name ilmart2 \
      --rm \
      -i \
      -v REPLACE_ME_WITH_THE_PATH_TO_ILMART_REPOSITORY:/code/ \
      -v REPLACE_ME_WITH_THE_PATH_TO_THE_DATA_FOLDER:/data/ \  
      -v REPLACE_ME_WITH_THE_PATH_TO_THE_RANKEVAL_DATA:/rankeval_data \
      -t ilmart:v2.0 bash
```

Then, you can run the experiments using the commands described in the README of each subfolder.
An improved documentation will be available upon request.
All the models and CSV used in the notebooks are available at this [link](https://drive.google.com/file/d/1fuv42ASlgIFCx624hJ2QajWUYCyikxNR/view?usp=sharing).

## Citation
If you use our work, please consider to cite it with:

```
@inproceedings{10.1145/3477495.3531840,
author = {Lucchese, Claudio and Nardini, Franco Maria and Orlando, Salvatore and Perego, Raffaele and Veneri, Alberto},
title = {ILMART: Interpretable Ranking with Constrained LambdaMART},
year = {2022},
isbn = {9781450387323},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3477495.3531840},
doi = {10.1145/3477495.3531840},
abstract = {Interpretable Learning to Rank (LtR) is an emerging field within the research area of explainable AI, aiming at developing intelligible and accurate predictive models. While most of the previous research efforts focus on creating post-hoc explanations, in this paper we investigate how to train effective and intrinsically-interpretable ranking models. Developing these models is particularly challenging and it also requires finding a trade-off between ranking quality and model complexity. State-of-the-art rankers, made of either large ensembles of trees or several neural layers, exploit in fact an unlimited number of feature interactions making them black boxes. Previous approaches on intrinsically-interpretable ranking models address this issue by avoiding interactions between features thus paying a significant performance drop with respect to full-complexity models. Conversely, ILMART, our novel and interpretable LtR solution based on LambdaMART, is able to train effective and intelligible models by exploiting a limited and controlled number of pairwise feature interactions. Exhaustive and reproducible experiments conducted on three publicly-available LtR datasets show that ILMART outperforms the current state-of-the-art solution for interpretable ranking of a large margin with a gain of nDCG of up to 8\%.},
booktitle = {Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {2255–2259},
numpages = {5},
keywords = {interpretable boosting, interpretable ranking, lambdamart},
location = {<conf-loc>, <city>Madrid</city>, <country>Spain</country>, </conf-loc>},
series = {SIGIR '22}
}
```