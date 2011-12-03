
#include "filereader.h"
#include "textlocation.h"
#include "recognition.h"

void singleTest(string fileName); // single image test
void batchTest();                 // batch test
void genTrainData();              // generate feature data for training
void printTextOnImg(CvArr* Img, const string text, CvPoint org);

int main (int argc, const char **argv) {

	string inputFile(argv[1]);
	singleTest(inputFile);
	//batchTest();
	//genTrainData();

	return 1;

}

void singleTest(string fileName) {

	// Read file
	FileReader fileReader(0, 0);
	fileReader.readFile(fileName);
	IplImage* loadImg = fileReader.scaledImg;

	// Preprocessing
	int imgWidth  = 480;
	int imgHeight = 108;
	TextLocation txt(imgWidth, imgHeight);
	txt.init(loadImg);
	txt.detectBlack(loadImg, txt.blackPixelImg);
	txt.thresholdImg(txt.grayImg, 2);

	// Number locating
	bool locateSuccess = txt.locateRecogBox();

	if (locateSuccess) {

		drawRects(txt.outImg, txt.segRects, CV_RGB(255,255,0));

		// Recognition
		Recognition recognizer;
		recognizer.initSvmModel("./");
		recognizer.recognize(txt.forRecogImg, txt.segRects);

		printTextOnImg(txt.outImg, recognizer.recognizedNum, cvPoint(20, 40));

	} else {

		printTextOnImg(txt.outImg, "fail box!", cvPoint(20, 40));

	}

	// Event loops
	while (1) {

		cvShowImage("box", txt.outImg);

		// Keyboard event
		int c = cvWaitKey(30);

		if ((char) c == 27) { // 'Esc' to terminate

			cvDestroyAllWindows();
			exit(1);

			break;
		}

	}

}

void batchTest() {

	ifstream inFile("./InvoiceDataSet0523/InvoiceDataSet0523-ans.txt");

	int imgWidth = 479;
	int imgHeight = 160;
	TextLocation txt(imgWidth, imgHeight);

	Recognition recognizer;
	recognizer.initSvmModel("./");

	string imgName;
	string ans;
	while (inFile >> imgName) {
		if (inFile >> ans) {

			cout << "processing: " << imgName << endl;

			FileReader fileReader(0, 0);
			string imgFile = "./InvoiceDataSet0523/" + imgName;
			fileReader.readFile(imgFile);
			IplImage* loadImg = fileReader.scaledImg;

			txt.init(loadImg);
			txt.detectBlack(loadImg, txt.blackPixelImg);
			txt.thresholdImg(txt.grayImg, 2);

			bool locateSuccess = txt.locateRecogBox();


			if (locateSuccess) { // bounding box is located correctly

				drawRects(txt.outImg, txt.segRects, CV_RGB(255,255,0));

				recognizer.recognize(txt.forRecogImg, txt.segRects);

				if (recognizer.recognizedNum == ans) { // Correct

					printTextOnImg(txt.outImg, recognizer.recognizedNum, cvPoint(20, 40));

					string savePath = "./InvoiceDataSet0523-testGood/" + imgName;
					cvSaveImage(savePath.c_str(), txt.outImg);

					/*savePath = "./i9000-near-testGood/" + imgName + "_src";
					cvSaveImage(savePath.c_str(), txt.forRecogImg);*/

				} else { // Wrong

					printTextOnImg(txt.outImg, recognizer.recognizedNum, cvPoint(20, 40));

					string savePath = "./InvoiceDataSet0523-testBad/" + imgName;
					cvSaveImage(savePath.c_str(), txt.outImg);

					/*savePath = "./InvoiceDataSet0523-testBad/" + imgName + "_src";
					cvSaveImage(savePath.c_str(), txt.forRecogImg);*/

				}

			} else {

				string savePath = "./InvoiceDataSet0523-testFailBox/" + imgName;
				cvSaveImage(savePath.c_str(), txt.outImg);

				/*savePath = "./InvoiceDataSet0523-testFailBox/" + imgName + "_EdgeBlack";
				cvSaveImage(savePath.c_str(), txt.m_edgeAndBlackImg);
				savePath = "./InvoiceDataSet0523-testFailBox/" + imgName + "_AdvEdge";
				cvSaveImage(savePath.c_str(), txt.m_advancedEdgeImg);*/


			}



		}

	}

	inFile.close();
}

void genTrainData() {

	ifstream inFile("./i9000-near/i9000-near-ans.txt");

	string imgName;
	string ans;

	int imgWidth = 479;
	int imgHeight = 160;
	TextLocation txt(imgWidth, imgHeight);

	Recognition recognizer;

	while (inFile >> imgName) {
		if (inFile >> ans) {

			cout << "processing: " << imgName << endl;
			cout << "ans: " << ans << endl;

			FileReader fileReader(0, 0);
			string imgFile = "./i9000-near/" + imgName;
			fileReader.readFile(imgFile);
			IplImage* loadImg = fileReader.scaledImg;

			txt.init(loadImg);
			txt.detectBlack(loadImg, txt.blackPixelImg);
			txt.thresholdImg(txt.grayImg, 2);

			bool locateSuccess = txt.locateRecogBox();

			if (locateSuccess) {

				drawRects(txt.outImg, txt.segRects, CV_RGB(255,255,0));

				recognizer.train(txt.forRecogImg, fileReader.extractFilename(imgName), ans, txt.segRects);

				string savePath = "./i9000-near-pick/" + imgName;
				cvSaveImage(savePath.c_str(), txt.outImg);

			} else {

				string savePath = "./i9000-near-fail/" + imgName;
				cvSaveImage(savePath.c_str(), txt.outImg);

			}

		}

	}


	inFile.close();

}

void printTextOnImg(CvArr* Img, const string text, CvPoint org) {

	CvFont txtFont = cvFont(3, 2);
	CvScalar txtColor = CV_RGB(255,0,0);
	cvPutText(Img, text.c_str(), org, &txtFont, txtColor);

}

