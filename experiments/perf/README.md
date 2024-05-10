# Performance evaluation

*Note:* for this experiments we used a private version of quickscorer, you might have to edit the scripts to make
them work.
We share this scripts just as an example of how to evaluate the performance the models.

Command used:

```bash
# Compile the C++ versione of ilmart scorer
cd /code/src/ilmart/fast_distilled
mkdir build
cd build
cmake ..
make

# Go back to this folder
cd /code/experiments/perf/
# Preprocess the data
python preproc.py
# Evaluate the scoring time
python scoring_time.py
```

