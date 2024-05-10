# ILMART experiments

In this folder are stored the main scripts to replicate the results described in the related paper for ILMART.
The following scripts commands have been used inside a docker container run as described in the main README.md file.

## Structure of the folder

- `basic` contains the python scripts for evaluating the effectiveness of ilmart by varying the number of main and interaction effects
- `fine-tuning` contains the python scripts for fine tuning ILMART using OPTUNA
- `notebooks` contains the jupyter notebooks to visualize the results and create the plots used in the paper

## Effectiveness varying main and interaction effects

To evaluate the effectiveness of ILMART by varying the number of main and interaction effects you can simply run the following commands:

```bash
cd basic
./no_inter_train_exp.sh /data
./inter_train_exp.sh /data
python eval_no_inter.py /data/no-inter/ /data/no-inter/eval.csv
python eval_inter.py /data/inter/ /data/inter/eval.csv
```

The results in terms of nDCG will be saved in the `/data/no-inter/eval.csv` and `/data/inter/eval.csv` files.

## Fine tuning ILMART

To fine tune ILMART using OPTUNA you can simply run the following commands:

```bash
cd fine-tuning
./no_inter_ft.sh /data/ft/
./inter_ft.sh /data/ft/
python eval.py
```

The results in terms of nDCG will be saved in the `/data/ft/eval.csv` file.