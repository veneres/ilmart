import copy
import itertools

import lightgbm as lgbm
import rankeval
from .ilmart_distill import IlmartDistill
from .utils import is_interpretable


def get_trees_contribution(model_to_rank: lgbm.Booster, pairs=False):
    distilled = IlmartDistill(model_to_rank)
    distill_contrib = [[feats, value] for feats, value in distilled.expected_contribution().items() if
                       len(feats) == (2 if pairs else 1)]
    distill_contrib.sort(key=lambda x: abs(x[1]), reverse=True)
    return distill_contrib


class Ilmart():

    def __init__(self,
                 lgbm_params: dict,
                 verbose=True,
                 main_effect_model_file=None,
                 feat_inter_boosting_rounds=2000,
                 n_inter_effects=50,
                 main_strategy="greedy",
                 inter_strategy="greedy"):
        self.verbose = verbose
        self.n_inter_effects = n_inter_effects
        valid_strategies = ["greedy", "prev", "contrib"]
        if main_strategy not in valid_strategies:
            raise ValueError(f"Main strategy must be a value in {valid_strategies}")
        if inter_strategy not in valid_strategies:
            raise ValueError(f"Inter strategy must be a value in {valid_strategies}")
        self.inter_strategy = inter_strategy
        self.main_strategy = main_strategy
        self.feat_inter_boosting_rounds = feat_inter_boosting_rounds
        self.lgbm_params = lgbm_params

        # preset all the default setting (no fitting)
        self._model_main_effects = None
        self._model_inter = None
        self._fit = False
        self.inter_rank = None

        # If a model is passed fixit as a main effect part
        if main_effect_model_file is not None:
            self._model_main_effects = lgbm.Booster(model_file=main_effect_model_file)
            self._model_inter = None
            self._fit = True

    def fit(self,
            num_boosting_rounds: int,
            train: rankeval.dataset.dataset.Dataset,
            vali: rankeval.dataset.dataset.Dataset,
            num_interactions=50,
            early_stopping_rounds=100,
            feat_imp: [int] = None) -> None:

        self.fit_main_effects(num_boosting_rounds, train, vali, early_stopping_rounds, feat_imp=feat_imp)
        if num_interactions > 0:
            self.fit_inter_effects(num_boosting_rounds,
                                   train,
                                   vali,
                                   num_interactions,
                                   early_stopping_rounds,
                                   feat_imp=feat_imp)

    def _add_verbose_log_callback(self, callbacks: list) -> None:
        if self.verbose:
            callbacks.append(lgbm.log_evaluation(period=1, show_stdv=True))

    def _add_early_stopping_callback(self, callbacks, early_stopping_rounds) -> None:
        early_stopping = lgbm.early_stopping(early_stopping_rounds, verbose=self.verbose)
        callbacks.append(early_stopping)

    def _get_main_effects_prev_params(self,
                                      feat_importances: list,
                                      n_main_effects: int = 0) -> dict:
        new_lgbm_params = copy.deepcopy(self.lgbm_params.copy())

        new_lgbm_params["tree_interaction_constraints"] = [[feat] for feat in feat_importances[:n_main_effects]]

        return new_lgbm_params

    def _get_main_effects_greedy_params(self, n_main_effects: int = 0) -> dict:
        new_lgbm_params = copy.deepcopy(self.lgbm_params.copy())

        new_lgbm_params["max_tree_interactions"] = 1
        if n_main_effects != 0:
            new_lgbm_params["max_interactions"] = n_main_effects

        return new_lgbm_params

    def _get_main_effects_contrib_params(self,
                                         train_lgbm: rankeval.dataset.dataset.Dataset,
                                         vali_lgbm: rankeval.dataset.dataset.Dataset,
                                         n_main_effects: int,
                                         contrib_boosting_rounds: int) -> dict:
        lgbm_params_contrib = self._get_main_effects_greedy_params(n_main_effects)
        lgbm_params_contrib["num_leaves"] = 3
        lgbm_params_contrib["learning_rate"] = 0.1
        model_contrib = lgbm.train(lgbm_params_contrib,
                                   train_lgbm,
                                   num_boost_round=contrib_boosting_rounds,
                                   valid_sets=[vali_lgbm])

        new_lgbm_params = copy.deepcopy(self.lgbm_params)

        # check the contribution of the first contrib_boosting_rounds trees
        contib_scores = get_trees_contribution(model_contrib, pairs=False)
        mif, scores = zip(*contib_scores)
        n_mif_list = mif[:n_main_effects]
        new_lgbm_params["tree_interaction_constraints"] = [[feat[0]] for feat in n_mif_list]

        return new_lgbm_params

    def fit_main_effects(self,
                         num_boosting_rounds: int,
                         train: rankeval.dataset.dataset.Dataset,
                         vali: rankeval.dataset.dataset.Dataset,
                         n_main_effects: int = 50,
                         early_stopping_rounds=100,
                         feat_imp: list = None,
                         contrib_boosting_rounds=None) -> None:

        train_lgbm = lgbm.Dataset(train.X, group=train.get_query_sizes(), label=train.y, free_raw_data=False)
        vali_lgbm = lgbm.Dataset(vali.X, group=vali.get_query_sizes(), label=vali.y, free_raw_data=False)

        if self.main_strategy == "greedy":
            new_lgbm_params = self._get_main_effects_greedy_params(n_main_effects)

        elif self.main_strategy == "prev":
            if feat_imp is None:
                raise ValueError("You must the precomputed feature importance (feat_imp) to use the strategy \"prev\"")
            new_lgbm_params = self._get_main_effects_prev_params(feat_imp, n_main_effects)

        else:  # Must be self.main_strategy == "contrib" since we already check inside the constructor
            if contrib_boosting_rounds is None:
                raise ValueError("You must the value of contrib_boosting_rounds to use the strategy \"contrib\"")
            new_lgbm_params = self._get_main_effects_contrib_params(train_lgbm,
                                                                    vali_lgbm,
                                                                    n_main_effects,
                                                                    contrib_boosting_rounds)

        callbacks = []

        self._add_verbose_log_callback(callbacks)
        self._add_early_stopping_callback(callbacks, early_stopping_rounds)

        if self.verbose:
            print(new_lgbm_params)

        self._model_main_effects = lgbm.train(new_lgbm_params,
                                              train_lgbm,
                                              num_boost_round=num_boosting_rounds,
                                              valid_sets=[vali_lgbm],
                                              callbacks=callbacks)

        self._fit = True

    def _get_main_effects_mif(self) -> list[int]:
        feat_importances = [(feat_ids, value) for feat_ids, value in
                            enumerate(self._model_main_effects.feature_importance("gain"))]
        feat_importances.sort(key=lambda x: x[1], reverse=True)
        most_important_feat = [feat_id for feat_id, value in feat_importances if value > 0]
        return most_important_feat

    def _get_inter_effects_greedy_params(self, n_inter_effects: int = 0):
        new_lgbm_params = copy.deepcopy(self.lgbm_params.copy())
        most_important_feats = self._get_main_effects_mif()
        new_lgbm_params["tree_interaction_constraints"] = [list(pair) for pair in
                                                           list(itertools.combinations(most_important_feats, 2))]
        if n_inter_effects != 0:
            new_lgbm_params["max_interactions"] = n_inter_effects

        return new_lgbm_params

    def _get_inter_effects_prev_params(self,
                                       feat_importances: list,
                                       n_inter_effects: int = 0) -> dict:
        new_lgbm_params = copy.deepcopy(self.lgbm_params.copy())
        main_effects_feat_used = self._get_main_effects_mif()
        most_important_feats = [feat for feat in feat_importances if feat in main_effects_feat_used]
        # Looping to find the right amount of main features to use. In this way we avoid problems of dealing with the
        # decimal part of computing n from (n*(n-1) )/ 2 = n_inter_effects
        comb_found = False
        comb_n_mif = None
        n = 2
        while not comb_found and n <= len(most_important_feats):
            comb_n_mif = list(itertools.combinations(most_important_feats[:n], 2))
            comb_found = len(comb_n_mif) > n_inter_effects
            n += 1
        if not comb_found:
            raise ValueError(f'Too few features to compute all the {n_inter_effects} most important pairs with the '
                             f'"prev" strategy.')

        new_lgbm_params["tree_interaction_constraints"] = [list(pair) for pair in comb_n_mif]

        if n_inter_effects != 0:
            new_lgbm_params["max_interactions"] = n_inter_effects

        return new_lgbm_params

    def _get_inter_effects_contrib_params(self,
                                          train: lgbm.Dataset,
                                          n_inter_effects: int,
                                          contrib_boosting_round: int) -> dict:
        max_feat_int = (self._model_main_effects.num_feature() * (self._model_main_effects.num_feature() - 1)) // 2
        lgbm_params = self._get_inter_effects_greedy_params(n_inter_effects=max_feat_int)
        lgbm_params["num_leaves"] = 3
        lgbm_params["learning_rate"] = 0.1

        if self.verbose:
            print(lgbm_params)

        model_to_rank = lgbm.train(lgbm_params,
                                   train,
                                   num_boost_round=contrib_boosting_round,
                                   init_model=self._model_main_effects)

        distilled_model = IlmartDistill(model_to_rank)

        pairs_n_contrib = [(feats, contrib) for feats, contrib in distilled_model.expected_contribution().items()
                           if len(feats) > 1]

        pairs_n_contrib.sort(key=lambda x: x[1], reverse=True)

        most_contrib_feats = [list(feats) for feats, contrib in pairs_n_contrib[:n_inter_effects]]

        new_lgbm_params = copy.deepcopy(self.lgbm_params.copy())

        new_lgbm_params["tree_interaction_constraints"] = most_contrib_feats

        return new_lgbm_params

    def fit_inter_effects(self,
                          num_boosting_rounds: int,
                          train: rankeval.dataset.dataset.Dataset,
                          vali: rankeval.dataset.dataset.Dataset,
                          num_interactions: int,
                          early_stopping_rounds=100,
                          contrib_boosting_rounds=1000,
                          feat_imp: [int] = None) -> None:
        self._check_fit()

        train_lgbm = lgbm.Dataset(train.X, group=train.get_query_sizes(), label=train.y, free_raw_data=False)
        vali_lgbm = lgbm.Dataset(vali.X, group=vali.get_query_sizes(), label=vali.y, free_raw_data=False)

        if self.inter_strategy == "greedy":
            new_lgbm_params = self._get_inter_effects_greedy_params(num_interactions)

        elif self.inter_strategy == "prev":
            if feat_imp is None:
                raise ValueError("You must the precomputed feature importance (feat_imp) to use the strategy \"prev\"")
            new_lgbm_params = self._get_inter_effects_prev_params(feat_imp, num_interactions)

        else:  # Must be self.inter_strategy == "contrib" since we already check inside the constructor
            if contrib_boosting_rounds is None:
                raise ValueError("You must the value of contrib_boosting_rounds to use the strategy \"contrib\"")
            new_lgbm_params = self._get_inter_effects_contrib_params(train_lgbm,
                                                                     num_interactions,
                                                                     contrib_boosting_rounds)

        early_stopping = lgbm.early_stopping(early_stopping_rounds, verbose=True)
        callbacks = [early_stopping]
        if self.verbose:
            callbacks.append(lgbm.log_evaluation(period=1, show_stdv=True))

        if self.verbose:
            print(new_lgbm_params)

        self._model_inter = lgbm.train(new_lgbm_params,
                                       train_lgbm,
                                       num_boost_round=num_boosting_rounds,
                                       valid_sets=[vali_lgbm],
                                       callbacks=callbacks,
                                       init_model=self._model_main_effects)

    def _check_fit(self) -> None:
        if self._model_main_effects is None:
            raise Exception("Model not fit yet")

    def get_model(self) -> lgbm.Booster:
        if self._model_inter is not None:
            return self._model_inter
        return self._model_main_effects

    def set_model(self, model: lgbm.Booster, inter=False) -> None:
        if not is_interpretable(model, verbose=False):
            raise Exception("The model passed is not interpretable")
        if not inter:
            self._model_main_effects = model
        else:
            self._model_inter = model

    def get_distill(self) -> IlmartDistill:
        if self._model_inter is not None:
            return IlmartDistill(self._model_inter)
        return IlmartDistill(self._model_main_effects)
