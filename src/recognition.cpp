
#include "../include/recognition.h"

Recognition::Recognition() :
    kClassCount(10),
	kfeatureXLength(40),
	kfeatureYLength(75) {

	recognizedNum = "";
	m_numberScore = new double[kClassCount];
	m_svmPredictor = NULL;

}

Recognition::Recognition(const vector<CvRect> &segRects) :
    kClassCount(10),
	kfeatureXLength(40),
	kfeatureYLength(75) {

	recognizedNum = "";
	m_numberScore = new double[kClassCount];
	m_svmPredictor = NULL;

}

void Recognition::initSvmModel(const string &dir) {

	m_svmPredictor = new SvmPredict(dir, (kfeatureXLength+kfeatureYLength));

}

void Recognition::train(IplImage* img, string fileName, string ans, const vector<CvRect> &segRects) {

	string filePath = "./i9000-near-pick-feature/";
	filePath.append(fileName);
	ofstream trainFile(filePath.c_str());

	if (trainFile.is_open()) {

		for (unsigned int i = 0; i < segRects.size(); ++i) {

			// get every single number's feature
			CvRect rect = segRects[i];
			IplImage* segRectImg = cvCreateImage(cvSize(rect.width, rect.height), img->depth,
					img->nChannels);
			getSubImg(img, rect, segRectImg);

			vector<double> feaVctor(kfeatureXLength + kfeatureYLength, 0);
			getSingleNumFeature(segRectImg, feaVctor);

#ifdef DEBUG
			cvShowImage("seg", segRectImg);
#endif

			cvReleaseImage(&segRectImg);

			// output the feature vector to .txt file
			trainFile << ans.substr(i, 1) << " ";
			for (unsigned int j = 0; j < feaVctor.size(); ++j) {
				trainFile << j + 1 << ":" << feaVctor[j] << " ";
			}
			trainFile << "\n";

		}

		trainFile.close();

	}

}

void Recognition::recognize(IplImage* img, const vector<CvRect> &segRects) {

	ostringstream oss;
	for (unsigned int i = 0; i < segRects.size(); ++i) {

		// get every single number's feature
		CvRect rect = segRects[i];
		IplImage* segRectImg = cvCreateImage(cvSize(rect.width, rect.height), img->depth,
				img->nChannels);

		getSubImg(img, rect, segRectImg);

		vector<double> feaVctor(kfeatureXLength + kfeatureYLength, 0);
		getSingleNumFeature(segRectImg, feaVctor);

		double formatFeaVctor[feaVctor.size()];
		vectorToArray(feaVctor, formatFeaVctor);

		m_svmPredictor->predict(formatFeaVctor, kfeatureXLength + kfeatureYLength, m_numberScore);

		int recognizedNum = outputRecognizedNum(m_numberScore);

		if (recognizedNum != -1)
			oss << recognizedNum;
		else
			oss << "?";

		cvReleaseImage(&segRectImg);

	}

	recognizedNum = oss.str();
	cout << "the answer is: " << recognizedNum << endl;

}

void Recognition::getNumberHistFeature(IplImage* edgeImg, vector<double> &feaVtor) {

	vector<int> histX(edgeImg->width, 0);
	for (int row = 0; row < edgeImg->height; row++) {

		uchar* pSrc = (uchar*) (edgeImg->imageData + row * edgeImg->widthStep);

		for (int col = 0; col < edgeImg->width; col++) {

			if (pSrc[col] == 255) {
				histX[col] += 1;
			}

		}
	}
	trimVector(histX);

	vector<int> histY(edgeImg->height, 0);
	for (int row = 0; row < edgeImg->height; row++) {

		uchar* pSrc = (uchar*) (edgeImg->imageData + row * edgeImg->widthStep);

		for (int col = 0; col < edgeImg->width; col++) {

			if (pSrc[col] == 255) {
				histY[row] += 1;
			}

		}
	}
	trimVector(histY);

	plot1DHisto(histX, 0, "histX");
	plot1DHisto(histY, 0, "histY");


	// interpolation feature to fixed length
	vector<int> featureX(kfeatureXLength, 0);
	vector<int> featureY(kfeatureYLength, 0);
	interpl(histX, featureX);
	interpl(histY, featureY);

	// normalize feature to range -1 ~ 1
	vector<double> norFeatureX(kfeatureXLength, 0);
	vector<double> norFeatureY(kfeatureXLength, 0);
	normalizeVector(featureX, norFeatureX);
	normalizeVector(featureY, norFeatureY);

	feaVtor = norFeatureX;
	feaVtor.insert( feaVtor.end(), norFeatureY.begin(), norFeatureY.end() );

}

void Recognition::interpl(const vector<int> &inVec, vector<int> &outVec) {

	int len = inVec.size(); // ��l�S�x�V�q���
	int newLength = outVec.size();
	for (int i = 0; i < newLength; ++i) {

		double interplPt = (double) (i) * (double) (len - 1) / (double) (newLength - 1);

		double upBound_d = ceil(interplPt);
		double lowBound_d = floor(interplPt);

		if ( upBound_d > (double) (len - 1) )
			upBound_d = (double) (len - 1);

		if ( lowBound_d > (double) (len - 1) )
			lowBound_d = (double) (len - 1);

		if (upBound_d == lowBound_d) {

			outVec[i] = inVec[(int)lowBound_d];

		} else {

			double x = upBound_d - lowBound_d;
			double delta;

			delta = inVec[(int) upBound_d] - inVec[(int) lowBound_d];
			outVec[i] = inVec[(int) lowBound_d] + (interplPt - lowBound_d) * (delta / x);

		}
	}

}

int Recognition::outputRecognizedNum(double* arrScore) {

	int maxProbIndex = 0;
	double maxProb = 0.0;
	for (int i = 0; i < kClassCount; ++i) {

		if (arrScore[i] > maxProb) {
			maxProb = arrScore[i];
			maxProbIndex = i;
		}
	}

	if (maxProb > 0.8)
		return maxProbIndex;
	else
		return -1;

}

void Recognition::getBinaryImgXHist(const IplImage* img, CvRect rect) {

	vector<int> hist(rect.width, 0);

	for (int row = rect.y; row < (rect.y + rect.height); row++) {

		uchar* pSrc = (uchar*) (img->imageData + row * img->widthStep);

		for (int col = rect.x; col < (rect.x + rect.width); col++) {

			if (pSrc[col] == 255) {
				hist[col - rect.x] += 1;
			}

		}
	}

	plot1DHisto(hist, 0, "x");

}

void Recognition::getBinaryImgYHist(const IplImage* img, CvRect rect) {

	vector<int> hist(rect.height, 0);

	for (int row = rect.y; row < (rect.y + rect.height); row++) {

		uchar* pSrc = (uchar*) (img->imageData + row * img->widthStep);

		for (int col = rect.x; col < (rect.x + rect.width); col++) {

			if (pSrc[col] == 255) {
				hist[row - rect.y] += 1;
			}

		}
	}

	trimVector(hist);

	plot1DHisto(hist, 0, "y");

}

void Recognition::normalizeVector(const vector<int> &feaVtor, vector<double> &norFeaVtor) {

	double minValue = *(min_element(feaVtor.begin(), feaVtor.end()));;
	double maxValue = *(max_element(feaVtor.begin(), feaVtor.end()));;

	double norMaxValue = 1.0;
	double norMinValue = -1.0;

	norFeaVtor.resize(feaVtor.size(), 0.0);

	for (unsigned int i = 0; i < norFeaVtor.size(); ++i) {

		double val = feaVtor[i];
		double norVal = 0.0;
		if (maxValue != minValue) {
			norVal = norMinValue + (val - minValue) * ((norMaxValue - norMinValue)
					/ (maxValue - minValue));
		} else {
			norVal = 0.0;
		}

		int dummy = (int)(norVal*100);
		norVal = (double)dummy / 100.0;

		norFeaVtor[i] = norVal;

	}

}

void Recognition::getSingleNumFeature(IplImage* img, vector<double> &feaVctor) {

	cvErode(img, img, NULL, 1);
	cvDilate(img, img, NULL, 2);

	IplImage* edgeImg = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
	cvCanny(img, edgeImg, 100, 300, 3);

	getNumberHistFeature(edgeImg, feaVctor);

	cvReleaseImage(&edgeImg);

}

void Recognition::vectorToArray(const vector<double> &vec, double* arr) {

	for (unsigned int i = 0; i < vec.size(); ++i) {
		arr[i] = vec[i];
	}
}

Recognition::~Recognition() {

	delete m_svmPredictor;
	delete[] m_numberScore;

}
