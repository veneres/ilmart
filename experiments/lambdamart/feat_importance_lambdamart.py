#!/usr/bin/env python
# coding: utf-8

import argparse
import lightgbm as lgbm
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Compute feature importance using lambdamart")
    parser.add_argument("models_path", type=str, help="Path to the folder containing the models")

    args = parser.parse_args()

    models_path = args.models_path

    models_path = Path(models_path)
    feat_imp_dict = {}
    for file in models_path.iterdir():
        if not file.is_file() or not file.name.endswith(".lgbm"):
            continue
        model = lgbm.Booster(model_file=file.resolve())
        feat_imp = [(feat_id, value) for feat_id, value in enumerate(model.feature_importance("gain"))]
        feat_imp.sort(key=lambda x: x[1], reverse=True)
        feat_sorted_by_imp = [feat_id for feat_id, feat_imp in feat_imp if feat_imp > 0]
        feat_imp_dict[file.name.replace(".lgbm", "")] = feat_sorted_by_imp

    for model_name, feat_imp in feat_imp_dict.items():
        print(f"Model: {model_name}")
        print(f"Feat importance: {feat_imp}")
        print("#" * 20)


if __name__ == '__main__':
    main()
