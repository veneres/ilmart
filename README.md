# ILMART: Interpretable Ranking with Constrained LambdaMART
This repository contains the proof of concept of ILMART, a modified version of LambdaMART to create an interpretable
learning-to-rank model.
Major details about the algorithms can be found in the related paper: TODO add link to the publication.

## Installation
There are two main steps to install the packages
1) Install a modified version of [LightGBM](https://github.com/veneres/LightGBM)
2) Install the sample package of ILMART

### Install the modified version of LightGBM:
The algorithm works with a modified version of [LightGBM](https://github.com/veneres/LightGBM). You can download it
as a submodule of this GIT repository cloning using:

```bash
git clone --recurse-submodules git://github.com/veneres/ilmart.git
```
Then, switch to the folder containing the installer of the python packaged:

```bash
cd LightGBM/python-package
```

Finally, run this command to install LightGBM (Linux and MacOS):

```bash
python setup.py install
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

Some simple scripts to replicate the major results presented in the related paper are available inside the 
[experiments folder](experiments).

## Citation
If you use our work, please consider to cite it with:

(To be generated)

