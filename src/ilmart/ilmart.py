import lightgbm as lgbm
import rankeval
from .ilmart_distill import IlmartDistill
from .utils import is_interpretable


class Ilmart():

    def __init__(self, verbose, feat_inter_boosting_rounds=2000, inter_rank_strategy="greedy"):
        self.verbose = verbose
        self._model_main_effects = None
        self._model_inter = None
        self.feat_inter_boosting_rounds = feat_inter_boosting_rounds
        self._fit = False
        self.inter_rank_strategy = inter_rank_strategy
        self.inter_rank = None

    def fit(self, lgbm_params: dict,
            num_boosting_rounds: int,
            train: rankeval.dataset.dataset.Dataset,
            vali: rankeval.dataset.dataset.Dataset,
            num_interactions=50, early_stopping_rounds=100):

        self.fit_main_effects(lgbm_params, num_boosting_rounds, train, vali, early_stopping_rounds)
        if num_interactions > 0:
            self.fit_inter_effects(lgbm_params, num_boosting_rounds, train, vali, num_interactions,
                                   early_stopping_rounds)

    def fit_main_effects(self,
                         lgbm_params: dict,
                         num_boosting_rounds: int,
                         train: rankeval.dataset.dataset.Dataset,
                         vali: rankeval.dataset.dataset.Dataset,
                         early_stopping_rounds=100):

        lgbm_params = lgbm_params.copy()
        train_lgbm = lgbm.Dataset(train.X, group=train.get_query_sizes(), label=train.y, free_raw_data=False)
        vali_lgbm = lgbm.Dataset(vali.X, group=vali.get_query_sizes(), label=vali.y, free_raw_data=False)

        lgbm_params["interaction_constraints"] = [[i] for i in range(train.n_features)]

        early_stopping = lgbm.early_stopping(early_stopping_rounds, verbose=True)
        callbacks = [early_stopping]

        if self.verbose:
            callbacks.append(lgbm.log_evaluation(period=1, show_stdv=True))

        if self.verbose:
            print(lgbm_params)

        self._model_main_effects = lgbm.train(lgbm_params,
                                              train_lgbm,
                                              num_boost_round=num_boosting_rounds,
                                              valid_sets=[vali_lgbm],
                                              callbacks=callbacks)
        self._fit = True

    def _get_contribution_greedy(self, model_to_rank: lgbm.Booster):
        tree_df = model_to_rank.trees_to_dataframe()
        greedy_contrib = []
        feat_name_to_index = {feat_name: feat_index for feat_index, feat_name in
                              enumerate(model_to_rank.feature_name())}
        for tree_index in tree_df["tree_index"].unique():
            # Compute feat used for tree with index tree:_index
            tree_df_per_index = tree_df[tree_df["tree_index"] == tree_index]
            feats_used = [feat_name_to_index[feat] for feat in tree_df_per_index["split_feature"].unique() if
                          feat is not None]
            feats_used = tuple(sorted(feats_used))
            if len(feats_used) < 2:
                continue
            if feats_used not in greedy_contrib:
                greedy_contrib.append(feats_used)
        return [list(pair) for pair in greedy_contrib]

    def _get_contribution_aware(self, model_to_rank: lgbm.Booster):
        distilled = IlmartDistill(model_to_rank)
        distill_contrib = [[feats, value] for feats, value in distilled.expected_contribution().items() if
                           len(feats) > 1]
        distill_contrib.sort(key=lambda x: abs(x[1]), reverse=True)
        return [list(feats) for feats, value in distill_contrib]

    def _get_contribution(self, model_to_rank: lgbm.Booster):
        if self.inter_rank_strategy == "greedy":
            return self._get_contribution_greedy(model_to_rank)
        else:
            return self._get_contribution_aware(model_to_rank)

    def _rank_interactions(self, lgbm_params, train: rankeval.dataset.dataset.Dataset):
        mif = [feat_i for feat_i, imp in enumerate(self._model_main_effects.feature_importance("split")) if imp > 0]
        not_mif = [feat for feat in range(train.n_features) if feat not in mif]

        # TODO do it without copying the entire dataset
        transformed_dataset = train.X.copy()
        transformed_dataset[:, not_mif] = 0

        train_lgbm = lgbm.Dataset(transformed_dataset, group=train.get_query_sizes(), label=train.y,
                                  free_raw_data=False)
        lgbm_params = lgbm_params.copy()
        lgbm_params["num_leaves"] = 3
        lgbm_params["learning_rate"] = 0.1

        if self.verbose:
            print(lgbm_params)

        model_to_rank = lgbm.train(lgbm_params,
                                   train_lgbm,
                                   num_boost_round=self.feat_inter_boosting_rounds,
                                   init_model=self._model_main_effects)

        return self._get_contribution(model_to_rank)

    def fit_inter_effects(self,
                          lgbm_params: dict,
                          num_boosting_rounds: int,
                          train: rankeval.dataset.dataset.Dataset,
                          vali: rankeval.dataset.dataset.Dataset,
                          num_interactions: int,
                          early_stopping_rounds=100,
                          force_inter_rank=False):
        self._check_fit()

        lgbm_params = lgbm_params.copy()

        if "interaction_constraints" in lgbm_params:
            lgbm_params.pop("interaction_constraints")

        train_lgbm = lgbm.Dataset(train.X, group=train.get_query_sizes(), label=train.y, free_raw_data=False)
        vali_lgbm = lgbm.Dataset(vali.X, group=vali.get_query_sizes(), label=vali.y, free_raw_data=False)

        if self.inter_rank is None or force_inter_rank:
            self.inter_rank = self._rank_interactions(lgbm_params, train)

        lgbm_params["tree_interaction_constraints"] = self.inter_rank[:num_interactions]

        if self.verbose:
            print(f"tree_interaction_constraints: {lgbm_params['tree_interaction_constraints']}")

        early_stopping = lgbm.early_stopping(early_stopping_rounds, verbose=True)
        callbacks = [early_stopping]
        if self.verbose:
            callbacks.append(lgbm.log_evaluation(period=1, show_stdv=True))

        if self.verbose:
            print(lgbm_params)

        self._model_inter = lgbm.train(lgbm_params,
                                       train_lgbm,
                                       num_boost_round=num_boosting_rounds,
                                       valid_sets=[vali_lgbm],
                                       callbacks=callbacks,
                                       init_model=self._model_main_effects)

    def _check_fit(self):
        if self._model_main_effects is None:
            raise Exception("Model not fit yet")

    def get_model(self):
        if self._model_inter is not None:
            return self._model_inter
        return self._model_main_effects

    def set_model(self, model: lgbm.Booster, inter=False):
        if not is_interpretable(model, verbose=False):
            raise Exception("The model passed is not interpretable")
        if not inter:
            self._model_main_effects = model
        else:
            self._model_inter = model

    def get_distill(self):
        if self._model_inter is not None:
            return IlmartDistill(self._model_inter)
        return IlmartDistill(self._model_main_effects)
