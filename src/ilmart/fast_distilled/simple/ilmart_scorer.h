//
// Created by Alberto Veneri on 27/03/23.
//

#ifndef FAST_DISTILLED_ILMART_SCORER_H
#define FAST_DISTILLED_ILMART_SCORER_H

#include <vector>
#include <unordered_map>
#include "../utilities.h"


class IlmartScorerVector {
public:
    explicit IlmartScorerVector(const std::string &model_path);

    double score(unsigned long row, DenseDataset &ds);

    std::vector<double> score_dataset(DenseDataset &ds);

    std::vector<std::vector<double>> splits;

    std::vector<std::vector<double>> splits_min;
    std::vector<std::vector<double>> splits_max;

    unsigned long n_features;

    // main effects scores
    std::vector<std::vector<double>> me_scores;
    std::vector<std::vector<double>> approximate_me_scores;

    std::vector<unsigned long> me_feats;

    // interaction effects scores
    std::vector<std::vector<double>> ie_scores;
    std::vector<std::vector<double>> approx_ie_scores;
    std::vector<std::pair<unsigned long, unsigned long>> ie_feats;


};

#endif //FAST_DISTILLED_ILMART_SCORER_H
