
#ifndef _RECOGNITION_H_
#define _RECOGNITION_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <core/core_c.h>
#include <highgui/highgui_c.h>
#include "utility.h"
#include "svm-predict.h"
using namespace std;


class Recognition {
public:

	Recognition();
	Recognition(const vector<CvRect> &segRects);

	string recognizedNum;

	void initSvmModel(const string &dir);
	void train(IplImage* img, string fileName, string ans, const vector<CvRect> &segRects);
	void recognize(IplImage* img, const vector<CvRect> &segRects);

	~Recognition();

private:

	const int kClassCount;
	const int kfeatureXLength;
	const int kfeatureYLength;

	double* m_numberScore;
	SvmPredict* m_svmPredictor;
	vector<CvRect> m_segRects;

	int outputRecognizedNum(double* arrScore);

	void getBinaryImgXHist(const IplImage* img, CvRect rect);
	void getBinaryImgYHist(const IplImage* img, CvRect rect);
	void interpl(const vector<int> &inVec, vector<int> &outVec);
	void getNumberHistFeature(IplImage* edgeImg, vector<double> &feaVtor);
	void normalizeVector(const vector<int> &feaVtor, vector<double> &norFeaVtor);
	void getSingleNumFeature(IplImage* img, vector<double> &feaVctor);
	void vectorToArray(const vector<double> &vec, double* arr);

};

#endif
