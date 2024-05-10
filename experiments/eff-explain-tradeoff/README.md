# Interpret Explain Tradeoff experiments
The following scripts have been used inside a docker container run as described in the main README.md file.

Commands used to train the models:

```bash
python main.py /data/tradeoff # with default parameters
./train.sh /data/tradeoff/ft # with fine tuning
```

Commands used to evaluate the models:

```bash
python eval.py /data/tradeoff /data/tradeoff/eval.csv # with default parameters
python eval.py /data/tradeoff/ft /data/tradeoff/ft/eval.csv # with fine tuning
python stat_sign_test.py # to perform the statistical significance test only on the fine tuned models
``` 