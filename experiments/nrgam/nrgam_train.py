import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_ranking as tfr
import argparse
import pickle
from pathlib import Path

tf.config.threading.set_inter_op_parallelism_threads(40)
tf.config.threading.set_intra_op_parallelism_threads(40)

DATSET_DICT = {
    "mslr_web/30k_fold1": "web30k",
    "istella/s": "istella",
    "yahoo": "yahoo_dataset"
}
LOG_NORMALIZATION = {
    "mslr_web/30k_fold1": False,
    "istella/s": True,
    "yahoo": False
}

LEARNING_RATE = 0.05
HIDDEN_LAYERS = [16, 8]
BATCH_SIZE = 128
# The minimum value in istella for 4 features (50, 134, 148, 176) could be slightly less than 0,
# and to avoid numerical issue with the log1p transformation we added a constant value to each feature.
NORMALIZATION_CONSTANT = 10
LOSS = "approx_ndcg_loss"


def ds_transform(ds, log=False):
    ds = ds.map(
        lambda feature_map: {key: tf.where(value < 10 ** 6, value, 10 ** 6) for key, value in feature_map.items()})
    ds = ds.map(lambda feature_map: {
        "_mask": tf.ones_like(feature_map["label"], dtype=tf.bool),
        **feature_map
    })
    ds = ds.padded_batch(batch_size=BATCH_SIZE)
    ds = ds.map(lambda feature_map: (feature_map, tf.where(feature_map["_mask"], feature_map.pop("label"), -1.)))
    if log:
        ds = ds.map(
            lambda feature_map, label: (
                {key: value + NORMALIZATION_CONSTANT for key, value in feature_map.items() if key != "_mask"}, label))
        ds = ds.map(
            lambda feature_map, label: (
                {key: tf.math.log1p(value) for key, value in feature_map.items() if key != "_mask"}, label))
    else:
        ds = ds.map(
            lambda feature_map, label: ({key: value for key, value in feature_map.items() if key != "_mask"}, label))
    return ds


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


def train_eval(ds_train, ds_vali, ds_test, name, initial_model, epochs=2000, patience=100, base_dir="."):
    CHECKPOINTS_FOLDER = f"{name}_checkpoints"

    feat_names = list(list(ds_train.take(1))[0][0].keys())
    model = init_model(feat_names, initial_model)

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_metric/ndcg_10',
                                                               patience=patience,
                                                               mode="max",
                                                               restore_best_weights=True)
    Path(CHECKPOINTS_FOLDER).mkdir(parents=True, exist_ok=True)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINTS_FOLDER)

    history = model.fit(ds_train, epochs=epochs, validation_data=ds_vali,
                        callbacks=[early_stopping_callback, checkpoint_callback])

    model.save(f"{base_dir}/{name}_model")

    with open(f"{base_dir}/{name}_history.pickle", "wb") as f:
        pickle.dump(history.history, f)

    eval_dict = model.evaluate(ds_test, return_dict=True)

    with open(f"{base_dir}/{name}_eval_dict.pickle", "wb") as f:
        pickle.dump(eval_dict, f)


def main():
    parser = argparse.ArgumentParser(description="Train results of Neural Rank Gam.")
    parser.add_argument("dataset", metavar="dataset", type=str,
                        choices=["mslr_web/30k_fold1", "istella/s", "yahoo"],
                        help="""
                        Dataset to be used during training. 
                        Possible choice to replicate the results: 
                        - mslr_web/30k_fold1
                        - istella/s
                        - yahoo
                        """)
    parser.add_argument("-keep_training", default=None, type=str, help="Path of the model to continue to train")
    parser.add_argument("-base_dir", default="../best_models/nrgam", type=str,
                        help="Path of the model to continue to train")

    args = parser.parse_args()

    Path(args.base_dir).mkdir(parents=True, exist_ok=True)

    ds_train = ds_transform(tfds.load(args.dataset, split="train"), log=LOG_NORMALIZATION[args.dataset])
    ds_vali = ds_transform(tfds.load(args.dataset, split="vali"), log=LOG_NORMALIZATION[args.dataset])
    ds_test = ds_transform(tfds.load(args.dataset, split="test"), log=LOG_NORMALIZATION[args.dataset])
    train_eval(ds_train, ds_vali, ds_test, DATSET_DICT[args.dataset], args.keep_training, base_dir=args.base_dir)


if __name__ == '__main__':
    main()
