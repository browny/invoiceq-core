
#ifndef _TEXTLOCATION_H_
#define _TEXTLOCATION_H_

#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <cv.h>
#include <highgui.h>

using namespace std;

enum ThrImgMode {SIMPLE_EDGE, ADVANCED_EDGE};

class TextLocation {

public:

	TextLocation(int width, int height);

	ThrImgMode thrImgMode;
	vector<CvRect> segRects;

	IplImage* outImg;
	IplImage* grayImg;
	IplImage* forRecogImg;
	IplImage* blackPixelImg;
	IplImage* m_simpleEdgeImg;
	IplImage* m_advancedEdgeImg;
	IplImage* m_edgeAndBlackImg;

	void init(const IplImage* img);
	void detectBlack(const IplImage* src, IplImage* blackPixelImg);
	void thresholdImg(const IplImage* grayImg, int dilateDeg);
	bool locateRecogBox();

	~TextLocation();

private:

	const int ksmoothSize;
	const int kboxMaxNum;
	const int kboxAreaThCoeff;
	const int kcandidateBoxVarTh;
	const float kccPerimeter;
	const double kareaRatio;
	const double khistoRatio;
	const double kglobalThBias;
	const double kblackVarTh;
	const double kblackAvgTh;

	IplImage* m_globalThImg;
	IplImage* m_smoothImg;
	IplImage* m_copyForCC;
	IplImage* m_maskForCC;
	IplImage* m_utilityMaskImg;
	IplImage* m_utilityZeroImg;

	bool getBoundingBox(const IplImage* src, int &clusterNum, CvRect &box);
	void checkBox(IplImage* src, CvRect &box);
	void clearifyHist(vector<int> &hist, int th);
	void horCut(vector<int> &hist, CvRect &rect);
	void verCut(vector<int> &hist, CvRect &rect);
	bool verifyBox(const IplImage* src, int num, CvRect &box, CvRect &verifiedBox);


	void getSegLines(const IplImage* img, CvRect numBox, vector<int> &segCenter);
	void getSegRects(const IplImage* src, const vector<int> segLines, CvRect box, vector<CvRect> &segRects);

	void getEdgeImg(const IplImage* src, IplImage* edgeImg, int dilateDeg);
	void getGlobalThImg(const IplImage* src, double th, IplImage* globalThImg, int dilateDeg);
	double getThForGlobalThImg(const IplImage* src, double ratio);

	void removeSmallRect(vector<CvRect> &rects, vector<CvPoint> &centers);
	void removePillarRect(vector<CvRect> &rects, vector<CvPoint> &centers);
	void getUnitedRects(const vector<CvRect> &rects, CvRect &numberBox);
	void getHistValleyCenters(const vector<int> &hist, vector<int> &centers);
	void getHistZeroIntervals(const vector<int> &hist, vector<int> &centers, vector<int> &widths);

	void getMaskImgFromRects(const IplImage* src, const vector<CvRect> &rects, IplImage* dst);
	void maxNumLimitedConnectComponet(const IplImage* img, const int maxNum, vector<CvRect> &rects,
			vector<CvPoint> &centers);

};


#endif
