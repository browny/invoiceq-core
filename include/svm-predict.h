#ifndef _SVM_PREDICT_H_
#define _SVM_PREDICT_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include "svm.h"
using namespace std;


struct scale_para_str {
    int index;
    double bottom_value;
    double up_value;
};

class SvmPredict {
public:

    SvmPredict(const string &dir, int featureLen);

    void predict(double* input_feature, int len, double* prob_estimates);


    ~SvmPredict();


private:

    string filePathDir;

    int max_nr_attr;
    int predict_probability;

    double scale_bottom;
    double scale_up;
    int scale_para_str_size;

    svm_node* x;
    svm_model* model;
    scale_para_str* scale_arr;

    void exit_input_error(int line_num);
    void exit_with_help();
    void load_scale_para(const char *scale_file_name, int len);

};

#endif

