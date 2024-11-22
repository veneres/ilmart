//
// Created by Alberto Veneri on 27/03/23.
//
#include <fstream>
#include <sstream>
#include <iterator>
#include <limits>    // std::numeric_limits
#include "ilmart_scorer.h"

IlmartScorerVector::IlmartScorerVector(const std::string &model_path) {
    std::ifstream infile;
    infile.open(model_path);
    std::string line_aux;

    // Read the number of features used
    std::getline(infile, line_aux);

    n_features = std::stoul(line_aux);

    splits.resize(n_features);
    me_scores.resize(n_features);
    ie_scores.resize(n_features * n_features);

    // Read the number of histograms computed (i.e. the components)
    std::getline(infile, line_aux);

    unsigned long n_components_used = std::stoi(line_aux);
    // read the splits
    for (int i = 0; i < n_components_used; ++i) {
        // First: read the number of features used by the histogram
        unsigned n_feat_used;
        std::getline(infile, line_aux);
        n_feat_used = std::stoi(line_aux);
        // Then read the feature used
        std::vector<unsigned long> component_feats(n_feat_used);
        for (int j = 0; j < n_feat_used; ++j) {
            std::getline(infile, line_aux);
            component_feats[j] = std::stoul(line_aux);
        }

        // Second: read all the splitting values for each feature
        for (auto feat: component_feats) {
            bool update = true;
            if (!splits[feat].empty()) {
                update = false;
            }
            std::getline(infile, line_aux);
            std::istringstream iss(line_aux);
            std::vector<std::string> tokens{std::istream_iterator<std::string>{iss},
                                            std::istream_iterator<std::string>{}};
            for (auto token: tokens) {
                double value;
                if (std::equal(token.begin(), token.end(), "-inf")) {
                    value = -std::numeric_limits<double>::max();
                } else if (std::equal(token.begin(), token.end(), "inf")) {
                    value = std::numeric_limits<double>::max();
                } else {
                    value = std::stod(token);
                }

                if (update) { // only update the splits the first time you read them
                    splits[feat].push_back(value);
                }
            }
        }

        // Third: read all the score values

        std::getline(infile, line_aux);
        std::istringstream iss(line_aux);
        std::vector<std::string> tokens{std::istream_iterator<std::string>{iss},
                                        std::istream_iterator<std::string>{}};
        std::vector<unsigned long> indexes(n_feat_used);

        for (const auto &token: tokens) {
            double value = std::stod(token);

            if (component_feats.size() == 1) {
                me_scores[component_feats[0]].push_back(value);
            } else if (n_feat_used == 2) {
                unsigned long idx = component_feats[0] * n_features + component_feats[1];
                ie_scores[idx].push_back(value);
            } else {
                throw std::invalid_argument("Each components must have 1 or 2 features");
            }
        }

        if (component_feats.size() == 1) {
            me_feats.push_back(component_feats[0]);
        } else {
            ie_feats.emplace_back(component_feats[0], component_feats[1]);
        }
    }


}

double IlmartScorerVector::score(unsigned long row, DenseDataset &ds) {
    double score = 0;
    // score the main effects and store buckets
    std::vector<unsigned long> buckets(ds.n_features);
    for (unsigned long feat_id: me_feats) {
        double feat_value = ds.read_feat_value(row, feat_id);
        auto it_lower_bound = std::lower_bound(splits[feat_id].begin(), splits[feat_id].end(), feat_value);
        unsigned bucket_id = std::distance(splits[feat_id].begin(), it_lower_bound) - 1;
        score += me_scores[feat_id][bucket_id];
        buckets[feat_id] = bucket_id;
    }

    for (auto [feat_id_1, feat_id_2]: ie_feats) {
        unsigned long bucket_id_1 = buckets[feat_id_1];
        unsigned long bucket_id_2 = buckets[feat_id_2];
        unsigned long vector_idx = bucket_id_1 * (splits[feat_id_2].size() - 1) + bucket_id_2;
        unsigned long key_id = feat_id_1 * n_features + feat_id_2;
        score += ie_scores[key_id][vector_idx];
    }
    return score;
}

std::vector<double> IlmartScorerVector::score_dataset(DenseDataset &ds) {
    std::vector<double> res(ds.n_samples);
    for (int i = 0; i < ds.n_samples; ++i) {
        res[i] = score(i, ds);
    }
    return res;
}

