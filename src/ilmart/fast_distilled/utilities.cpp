//
// Created by Alberto Veneri on 23/03/23.
//

#include "utilities.h"
#include <cstddef>
#include <cstdio>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iterator>
#include <vector>

// read in a problem (in svmlight format)
typedef unsigned long ulong;


DenseDataset::DenseDataset(ulong n_features, ulong n_samples) : n_features(n_features),
                                                                n_samples(n_samples),
                                                                _dense_matrix(n_features * n_samples),
                                                                labels(n_samples) {}

double DenseDataset::read_feat_value(unsigned long row, unsigned long feat) {
    return _dense_matrix[row * n_features + feat];
}

void DenseDataset::insert_value(unsigned long row, unsigned long feat, double value) {
    _dense_matrix[row * n_features + feat] = value;
}


DenseDataset read_dataset(const std::string &filename) {
    ulong n_rows = 0;
    ulong max_id_feat = 0;
    ulong n_features = 0;
    std::string line;

    // first read to get the number of lines and the number of features
    std::ifstream infile;
    infile.open(filename);
    if (!infile.is_open()) {
        std::cerr << "failed to open " << filename << '\n';
        exit(1);
    }
    while (std::getline(infile, line)) {
        n_rows += 1;
        std::istringstream iss(line);
        std::vector<std::string> tokens{std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{}};
        std::string last_token = tokens[tokens.size() - 1];
        max_id_feat = std::max((unsigned long) std::stoll(last_token.substr(0, last_token.find(':'))), max_id_feat);
    }
    infile.clear();
    infile.seekg(0, infile.beg);

    n_features = max_id_feat + 1;

    DenseDataset res = DenseDataset(n_features, n_rows);

    ulong sample_id = 0;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::vector<std::string> tokens{std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{}};

        res.labels[sample_id] = std::stoi(tokens[0]);

        for (int i = 2; i < tokens.size(); ++i) {
            auto token = tokens[i];
            auto separator_pos = token.find(':');
            ulong feat_id = (unsigned long) std::stoll(token.substr(0, separator_pos));
            double value = std::stod(token.substr(separator_pos + 1, token.size() - separator_pos));
            res.insert_value(sample_id, feat_id, value);
        }
        sample_id++;

    }
    infile.close();
    infile.clear();
    return res;
}