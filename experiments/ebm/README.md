# EBM experiments 
The following scripts have been used inside a docker container run as described in the main README.md file.

## Training
To train the model you can simply run the two scripts `no_inter_ft.sh` and `inter_ft.sh` to train the models without 
and with the interactions respectively. 
The script will train the model and saved them inside the `/data/ebm` folder.
```bash
./no_inter_ft.sh
./inter_ft.sh
```

## Evaluation
To evaluate the model you can simply run the `eval.py`.
The results will be saved inside the `/data/ebm/eval.csv` file.

```bash
python eval.py
```