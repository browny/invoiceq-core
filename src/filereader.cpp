
#include "../include/filereader.h"
#include <assert.h>
#include <iostream>
#include "../include/opencv/imgproc/imgproc_c.h"
using namespace std;

FileReader::FileReader(int argc, const char** argv) :
	kLongsideMax(640) {

	m_srcImg = 0;
	scaledImg = 0;

	if (argc < 2) {

	} else {

		cout << "> load image from: " << argv[1] << endl;

	}

}

bool FileReader::readFile(string filePath) {

	bool readOK = false;
	//dragFileName = filePath;

	m_srcImg = cvLoadImage(filePath.c_str());

	if (!m_srcImg) { // fail to load image

		readOK = false;
		return readOK;

	} else {

		scaleImg(*m_srcImg);

		readOK = true;
		return readOK;

	}
}

string FileReader::extractFilename(const string& path) {

	int start = path.find_last_of('\\') + 1;
	int end = path.find_last_of('.');

	return path.substr(start, end-start);
}

void FileReader::scaleImg(const IplImage &srcImg) {

	int longSide = (srcImg.width > srcImg.height) ? srcImg.width : srcImg.height;

	if (longSide > kLongsideMax) {

		double scaleRatio = (double) longSide / kLongsideMax;

		int newWidth = (double) (srcImg.width) / scaleRatio;
		int newHeight = (double) (srcImg.height) / scaleRatio;
		CvSize newSize = cvSize(newWidth, newHeight);
		scaledImg = cvCreateImage(newSize, srcImg.depth, srcImg.nChannels);

		cvResize(&srcImg, scaledImg);

	} else {

		scaledImg = cvCreateImage(cvGetSize(&srcImg), srcImg.depth, srcImg.nChannels);
		cvCopy(&srcImg, scaledImg);

	}

}

FileReader::~FileReader() {

	cvReleaseImage(&m_srcImg);
	cvReleaseImage(&scaledImg);

}
