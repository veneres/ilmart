//
// Created by Alberto Veneri on 23/03/23.
//

#include <iostream>
#include <fstream>
#include <chrono>
#include "utilities.h"
#include "simple/ilmart_scorer.h"

using namespace std;

int main(int argc, char **argv) {
    if (argc <= 4) {
        cout << "You must provide 4 arguments: <model path> <dataset path> <output path> <rep count>" << endl;
        exit(1);
    }
    string model_path = argv[1];
    string dataset_path = argv[2];
    string out_path = argv[3];
    int rep_count = std::stoi(argv[4]);
    cout << "Model path: " << model_path << endl;
    cout << "Dataset path: " << dataset_path << endl;

    DenseDataset test_ds = read_dataset(dataset_path);
    cout << "Dataset read" << endl;
    IlmartScorerVector ilmartsv(model_path);
    vector<std::chrono::milliseconds> elapsed_times;
    vector<double> res;
    for (int  i = 0;  i < rep_count; ++ i) {
        auto begin = std::chrono::high_resolution_clock::now();
        res = ilmartsv.score_dataset(test_ds);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
        elapsed_times.push_back(elapsed);
    }

    cout << "Number of trials: " << rep_count << endl;
    cout << "------------------" << endl;
    cout << "Elapsed times (ms): " << endl;
    for (auto elem: elapsed_times) {
        cout << elem.count() << endl;
    }
    cout << "------------------" << endl;
    cout << "Average time (ms): " << endl;
    long long sum = 0;
    for (auto elem: elapsed_times) {
        sum += elem.count();
    }
    cout << (double) sum / rep_count << endl;


    ofstream outfile;
    outfile.open(out_path);
    for (auto elem: res) {
        outfile << elem << endl;
    }

    return 0;
}