
#include <iostream>
#include "../include/svm-predict.h"

using namespace std;


SvmPredict::SvmPredict(const string &dir, int featureLen) {

    max_nr_attr = 0;
    predict_probability = 1;

    // parameter for scaling
    scale_bottom = 0;
    scale_up = 0;
    scale_para_str_size = 0;

    // load 'model'
    //string modelPath = dir + "model";
    string modelPath = "model";
    model = svm_load_model(modelPath.c_str());

    // load 'scaling parameter'
    string scaleParaPath = dir + "scale_para";
    load_scale_para(scaleParaPath.c_str(), featureLen);

}

void SvmPredict::exit_input_error(int line_num) {
    fprintf(stderr, "Wrong input format at line %d\n", line_num);
    exit(1);
}

void SvmPredict::exit_with_help() {
    printf(
            "Usage: svm-predict [options] test_file model_file output_file\n"
            "options:\n"
            "-b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0); for one-class SVM only 0 is supported\n");
    exit(1);
}

void SvmPredict::load_scale_para(const char *scale_file_name, int len) {

    FILE *fp = fopen(scale_file_name,"rb");

    char cmd[256];

    // first line
    fscanf(fp,"%50s",cmd);   

    // second line
    fscanf(fp,"%50s",cmd);   
    scale_bottom = atof(cmd);	
    fscanf(fp,"%50s",cmd);   
    scale_up = atof(cmd);

    // read scaling paramenter
    scale_para_str_size = 0;
    scale_arr = (struct scale_para_str *) malloc(len*sizeof(struct scale_para_str));
    while(fscanf(fp,"%50s",cmd)!=EOF){

        scale_arr[scale_para_str_size].index = atoi(cmd);

        fscanf(fp,"%50s",cmd);
        scale_arr[scale_para_str_size].bottom_value = atof(cmd);

        fscanf(fp,"%50s",cmd);
        scale_arr[scale_para_str_size].up_value = atof(cmd);

        scale_para_str_size++;
    }

}

void SvmPredict::predict(double* input_feature, int len, double* prob_estimates) {

    // initilaize the memory for feature
    max_nr_attr = len + 2;
    x = (struct svm_node *) malloc(max_nr_attr * sizeof(struct svm_node));

    // read feature into x, and do scaling
    int i;
    for (i = 0; i < scale_para_str_size; i++) {
        x[i].index = scale_arr[i].index;

        double input_element = input_feature[x[i].index - 1];
        x[i].value = (input_element - scale_arr[i].bottom_value) / (scale_arr[i].up_value
                - scale_arr[i].bottom_value) * (scale_up - scale_bottom) + scale_bottom;

    }
    x[i].index = -1;

    // predict the probability of each class
    svm_predict_probability(model, x, prob_estimates);

    free(x);

}

SvmPredict::~SvmPredict() {

    free(scale_arr);
    svm_destroy_model(model);

}

