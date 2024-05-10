//
// Created by Alberto Veneri on 23/03/23.
//

#ifndef FAST_DISTILLED_UTILITIES_H
#define FAST_DISTILLED_UTILITIES_H

#include <string>
#include <vector>
#include <unordered_map>

class DenseDataset {
public:
    DenseDataset(unsigned long n_features, unsigned long n_samples);

    void insert_value(unsigned long row, unsigned long feat, double value);

    double read_feat_value(unsigned long row, unsigned long feat);

    std::vector<unsigned> labels;

    unsigned long n_features;
    unsigned long n_samples;

private:
    std::vector<double> _dense_matrix;
};

DenseDataset read_dataset(const std::string &filename);



#endif //FAST_DISTILLED_UTILITIES_H
