# NeuralRankGAM experiments 
The following scripts have been used inside a docker container run as described in the main README.md file.

## Training
To train the model you can simply run the `train.sh` script. 
The script will train the model and saved them inside the `/data/nrgam` folder.
```bash
./train.sh /data/nrgam
```
## Evaluation
To evaluate the model you can simply run the `eval.py`.
The results will be saved inside the `/data/nrgam/eval.csv` file.

```bash
python eval.py
```