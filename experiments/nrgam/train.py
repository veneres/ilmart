import time

import tensorflow as tf
import tensorflow_datasets as tfds
import yahoo_dataset
import tensorflow_ranking as tfr
import argparse
import pickle
from pathlib import Path
from utils import ds_transform, LOG_NORMALIZATION

tf.config.threading.set_inter_op_parallelism_threads(100)


LEARNING_RATE = 0.05
HIDDEN_LAYERS = [16, 8]
PATIENCE = 100
EPOCHS = 3000

SAVE_EVERY = 200

LOSS = "approx_ndcg_loss"


def init_model(feat_names, initial_model):
    if initial_model is not None:
        model = tf.keras.models.load_model(initial_model)
        print(f"Model correctly loaded from {initial_model}")
    else:

        feat_cols = {name: tf.feature_column.numeric_column(name, shape=(1,), default_value=0.0)
                     for name in feat_names}

        network = tfr.keras.canned.GAMRankingNetwork(
            context_feature_columns=None,
            example_feature_columns=feat_cols,
            example_hidden_layer_dims=HIDDEN_LAYERS,
            activation=tf.nn.relu,
            use_batch_norm=True)

        loss = tfr.keras.losses.get(LOSS)
        metrics = tfr.keras.metrics.default_keras_metrics()
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=LEARNING_RATE)
        model = tfr.keras.model.create_keras_model(network=network,
                                                   loss=loss,
                                                   metrics=metrics,
                                                   optimizer=optimizer,
                                                   size_feature_name=None)

    return model


def train(ds_train, ds_vali, name, initial_model, epochs=EPOCHS, patience=PATIENCE, base_dir="."):
    CHECKPOINTS_FOLDER = f"{base_dir}/{name}_checkpoints"

    feat_names = list(list(ds_train.take(1))[0][0].keys())
    model = init_model(feat_names, initial_model)

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_metric/ndcg_10',
                                                               patience=patience,
                                                               mode="max",
                                                               restore_best_weights=True)
    Path(CHECKPOINTS_FOLDER).mkdir(parents=True, exist_ok=True)


    # Save the model every 10 epochs (len(ds_train) is the number of batches in the training set)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINTS_FOLDER, save_freq=len(ds_train) * SAVE_EVERY)
    # time the function below
    start_time = time.time()

    history = model.fit(ds_train, epochs=epochs, validation_data=ds_vali,
                        callbacks=[early_stopping_callback, checkpoint_callback])
    end_time = time.time()
    print(f"Training took {end_time - start_time} seconds, or {(end_time - start_time) / 60} minutes, or {(end_time - start_time) / 3600} hours.")

    model.save(f"{base_dir}/{name}_model")

    with open(f"{base_dir}/{name}_history.pickle", "wb") as f:
        pickle.dump(history.history, f)

    with open(f"{base_dir}/{name}_time.txt", "w") as f:
        f.write(f"{start_time}\n")
        f.write(f"{end_time}\n")
        f.write(f"{end_time - start_time}\n")


def main():
    parser = argparse.ArgumentParser(description="Train results of Neural Rank Gam.")
    parser.add_argument("dataset", metavar="dataset", type=str,
                        choices=["web30k", "istella", "yahoo"],
                        help="""
                        Dataset to be used during training. 
                        Possible choice to replicate the results: 
                        - web30k
                        - istella
                        - yahoo
                        """)
    parser.add_argument("--keep_training", default=None, type=str, help="Path of the model to continue to train")
    parser.add_argument("--base_dir", default="/data/nrgam", type=str,
                        help="Path where to save the trained models.")

    args = parser.parse_args()

    dataset_name = args.dataset

    Path(args.base_dir).mkdir(parents=True, exist_ok=True)

    ds_train = ds_transform(tfds.load(f"{dataset_name}_rankeval", split="train"), log=LOG_NORMALIZATION[args.dataset])
    ds_vali = ds_transform(tfds.load(f"{dataset_name}_rankeval", split="vali"), log=LOG_NORMALIZATION[args.dataset])
    train(ds_train, ds_vali, dataset_name, args.keep_training, base_dir=args.base_dir)


if __name__ == '__main__':
    main()
