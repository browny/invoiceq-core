
#ifndef _UTILITY_H_
#define _UITLITY_H_

#include <vector>
#include <algorithm>
#include <iostream>
#include <cv.h>
#include <highgui.h>
using namespace std;

double dist(CvPoint a, CvPoint b);

void inverseBinaryImage(IplImage* img);

void connectComponent(IplImage* src, const int poly_hull0, const float perimScale, int *num,
		vector<CvRect> &rects, vector<CvPoint> &centers);

void getSubImg(IplImage* src, const CvRect &roiRect, IplImage* subImg);

void plot1DHisto(const vector<int> &hist, int markIdx, string winName);

void getHorHistByRect(const IplImage* src, CvRect rect, vector<int> &hist);

void getVerHistByRect(const IplImage* src, CvRect rect, vector<int> &hist);

void trimVector(vector<int> &hist);

void drawRects(IplImage* img, vector<CvRect> &rects, CvScalar color);


#endif
