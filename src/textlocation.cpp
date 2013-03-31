
#include "../include/textlocation.h"
#include "../include/utility.h"

TextLocation::TextLocation(int width, int height) :
    ksmoothSize(3),
    kboxMaxNum(20),
    kboxAreaThCoeff(2),
    kcandidateBoxVarTh(1000),
    kccPerimeter(20.0f),
    kareaRatio(0.02),
    khistoRatio(0.8),
    kglobalThBias(-10),
    kblackVarTh(400),
    kblackAvgTh(80)
{

    thrImgMode = SIMPLE_EDGE;

    segRects.resize(0);

    CvSize imgSize = cvSize(width, height);

    grayImg = cvCreateImage(imgSize, IPL_DEPTH_8U, 1);
    blackPixelImg = cvCreateImage(imgSize, IPL_DEPTH_8U, 1);
    forRecogImg = cvCreateImage(imgSize, IPL_DEPTH_8U, 1);
    m_globalThImg = cvCreateImage(imgSize, IPL_DEPTH_8U, 1);
    m_simpleEdgeImg = cvCreateImage(imgSize, IPL_DEPTH_8U, 1);
    m_advancedEdgeImg = cvCreateImage(imgSize, IPL_DEPTH_8U, 1);
    m_edgeAndBlackImg = cvCreateImage(imgSize, IPL_DEPTH_8U, 1);
    m_smoothImg = cvCreateImage(imgSize, IPL_DEPTH_8U, 1);
    m_copyForCC = cvCreateImage(imgSize, IPL_DEPTH_8U, 1);
    m_maskForCC = cvCreateImage(imgSize, IPL_DEPTH_8U, 1);
    m_utilityMaskImg = cvCreateImage(imgSize, IPL_DEPTH_8U, 1);
    m_utilityZeroImg = cvCreateImage(imgSize, IPL_DEPTH_8U, 1);
    outImg = cvCreateImage(imgSize, IPL_DEPTH_8U, 3);


}

void TextLocation::init(const IplImage* img) {

    thrImgMode = SIMPLE_EDGE;
    segRects.clear();

    cvZero(grayImg);
    cvZero(blackPixelImg);
    cvZero(forRecogImg);
    cvZero(m_globalThImg);
    cvZero(m_simpleEdgeImg);
    cvZero(m_advancedEdgeImg);
    cvZero(m_edgeAndBlackImg);
    cvZero(m_smoothImg);
    cvZero(m_copyForCC);
    cvZero(m_maskForCC);
    cvZero(m_utilityMaskImg);
    cvZero(m_maskForCC);
    cvZero(outImg);

    cvCopy(img, outImg);
    cvCvtColor(img, grayImg, CV_RGB2GRAY);

}

void TextLocation::detectBlack(const IplImage* src, IplImage* blackPixelImg) {

    cvZero(blackPixelImg);

    double r = 0;
    double g = 0;
    double b = 0;
    double avg = 0;
    double var = 0;

    double rowStart = 0.0 * (double)src->height;
    double rowEnd   = 1.0 * (double)src->height;

    for (int row = rowStart; row < rowEnd; row++) {

        uchar* pSrc = (uchar*) (src->imageData + row * src->widthStep);
        uchar* pBlack = (uchar*) (blackPixelImg->imageData + row * blackPixelImg->widthStep);

        for (int col = 0; col < src->width; col++) {

            b = pSrc [3*col + 0];
            g = pSrc [3*col + 1];
            r = pSrc [3*col + 2];

            avg = (r+g+b) / 3.0;
            var = (avg-r)*(avg-r) + (avg-g)*(avg-g) + (avg-b)*(avg-b);

            if ( (var < kblackVarTh) && (avg < kblackAvgTh) ) {

                pBlack[col] = 255;

            }
        }
    }

    cvDilate(blackPixelImg, blackPixelImg, NULL, 3);
#ifdef DEBUG
    cvShowImage("blackPixelImg", blackPixelImg);
#endif

}

bool TextLocation::verifyBox(const IplImage* src, int num, CvRect &box, CvRect &verifiedBox) {

    bool haveGoodBox = false;

    if (num == 1) {

        vector<int> segCenters(0);
        getSegLines(src, box, segCenters);

        if (segCenters.size() == 9) {

            verifiedBox = box;
            haveGoodBox = true;
        }

    }

    return haveGoodBox;

}

void TextLocation:: checkBox(IplImage* src, CvRect &box) {


    vector<int> verHist(0);
    getVerHistByRect(src, box, verHist);
    clearifyHist(verHist, 40);
    plot1DHisto(verHist, verHist.size()/2, "box verhist");
    verCut(verHist, box);


    vector<int> horHist(0);
    getHorHistByRect(src, box, horHist);
    clearifyHist(horHist, 10);
    plot1DHisto(horHist, horHist.size() / 2, "box horhist");
    horCut(horHist, box);



}

void TextLocation::clearifyHist(vector<int> &hist, int th) {

    for (unsigned int i = 0; i < hist.size(); ++i) {
        if (hist[i] < th) {
            hist[i] = 0;
        }
    }
}

void TextLocation::horCut(vector<int> &hist, CvRect &rect) {

    vector<int> valleyCenters(0);
    vector<int> valleyWidths(0);
    getHistZeroIntervals(hist, valleyCenters, valleyWidths);

    /*cout << "width: ";
      for (unsigned int i = 0; i < valleyWidths.size(); ++i) {
      cout << valleyWidths[i] << " ";
      }
      cout << endl;*/


    if (valleyCenters.size() >= 9) {

        vector<int> sortedValleyWidths(valleyWidths);
        sort(sortedValleyWidths.begin(), sortedValleyWidths.end());

        int sum = 0;
        for (unsigned int i = 0; i < sortedValleyWidths.size()-2; ++i)
            sum += sortedValleyWidths[i];
        double avg = (double)sum / (double)(sortedValleyWidths.size()-2);

        int th = (int)(3.0 * avg);

        for (unsigned int i = 0; i < valleyWidths.size(); ++i) {

            int refCenter = 0;
            if (valleyWidths[i] > th) {

                if ( i < (valleyWidths.size()/2) ) {

                    rect.x = rect.x + (valleyCenters[i] - refCenter);
                    rect.width -= (valleyCenters[i] - refCenter);

                    refCenter = valleyCenters[i];

                } else {

                    rect.width -= hist.size() - valleyCenters[i];
                    rect.width -= (valleyWidths[i]/3);
                    break;

                }

            }

        }

    }

}

void TextLocation::verCut(vector<int> &hist, CvRect &rect) {

    vector<int> valleyCenters(0);
    vector<int> valleyWidths(0);
    getHistZeroIntervals(hist, valleyCenters, valleyWidths);

    /*cout << "width: ";
      for (unsigned int i = 0; i < valleyWidths.size(); ++i) {
      cout << valleyWidths[i] << " ";
      }
      cout << endl;*/

    if (valleyCenters.size() > 2) {

        int sum = 0;
        int choosenMountain = 0;
        for (unsigned int i = 0; i < valleyCenters.size() - 1; ++i) {

            int tempSum = 0;
            for (int j = valleyCenters[i]; j < valleyCenters[i + 1]; ++j) {
                tempSum += hist[j];
            }

            if (tempSum > sum) {
                sum = tempSum;
                choosenMountain = i;
            }
        }

        /*cout << "choosenMountain: " << choosenMountain << endl;*/

        rect.y = rect.y + valleyCenters[choosenMountain];
        rect.height = valleyCenters[choosenMountain + 1]
            - valleyCenters[choosenMountain];

    }

}


bool TextLocation::locateRecogBox() {

    // Verify edgeAndBlack image
    int clusterNum = 0;
    CvRect box;
    bool havebox = getBoundingBox(m_edgeAndBlackImg, clusterNum, box);
    if (havebox)
        checkBox(m_edgeAndBlackImg, box);

    // Draw boxes
    /*for (unsigned int i = 0; i < boxes.size(); ++i) {
      cvRectangle(
      outImg,
      cvPoint(boxes[i].x, boxes[i].y),
      cvPoint(boxes[i].x + boxes[i].width,
      boxes[i].y + boxes[i].height), CV_RGB(255, 0, 255), 3);
      }*/

    CvRect verifiedBox = cvRect(0, 0, 0, 0);

    bool edgeAndBlackGood = verifyBox(m_edgeAndBlackImg, clusterNum, box, verifiedBox);

    /*if (edgeAndBlackGood) {
      cvRectangle(
      outImg,
      cvPoint(verifiedBox.x, verifiedBox.y),
      cvPoint(verifiedBox.x + verifiedBox.width,
      verifiedBox.y + verifiedBox.height), CV_RGB(255, 0, 255), 3);
      }*/


    // Verify edge image
    int clusterNum2 = 0;
    CvRect box2;

    bool edgeGood = false;
    CvRect verifiedBox2 = cvRect(0, 0, 0, 0);

    if (thrImgMode == ADVANCED_EDGE) {

        bool havebox = getBoundingBox(m_advancedEdgeImg, clusterNum2, box2);

        if (havebox)
            checkBox(m_advancedEdgeImg, box2);

        edgeGood = verifyBox(m_advancedEdgeImg, clusterNum2, box2, verifiedBox2);

    }
    else if (thrImgMode == SIMPLE_EDGE) {

        bool havebox = getBoundingBox(m_simpleEdgeImg, clusterNum2, box2);

        if (havebox)
            checkBox(m_simpleEdgeImg, box2);

        edgeGood = verifyBox(m_simpleEdgeImg, clusterNum2, box2, verifiedBox2);
    }

    // Draw boxes2
    /*for (unsigned int i = 0; i < boxes2.size(); ++i) {
      cvRectangle(
      outImg,
      cvPoint(boxes2[i].x, boxes2[i].y),
      cvPoint(boxes2[i].x + boxes2[i].width,
      boxes2[i].y + boxes2[i].height), CV_RGB(0, 255, 255), 1);
      }*/

    /*if (edgeGood) {
      cvRectangle(
      outImg,
      cvPoint(verifiedBox2.x, verifiedBox2.y),
      cvPoint(verifiedBox2.x + verifiedBox2.width,
      verifiedBox2.y + verifiedBox2.height),
      CV_RGB(0, 255, 255), 3);
      }*/

    if (edgeAndBlackGood) {

        vector<int> segLines(0);
        getSegLines(m_edgeAndBlackImg, verifiedBox, segLines);
        getSegRects(m_edgeAndBlackImg, segLines, verifiedBox, this->segRects);

        return true;

    }
    else if (edgeGood) {

        if (thrImgMode == ADVANCED_EDGE) {

            vector<int> segLines(0);
            getSegLines(m_advancedEdgeImg, verifiedBox2, segLines);
            getSegRects(m_advancedEdgeImg, segLines, verifiedBox2,
                    this->segRects);

        } else if (thrImgMode == SIMPLE_EDGE) {

            vector<int> segLines(0);
            getSegLines(m_simpleEdgeImg, verifiedBox2, segLines);
            getSegRects(m_simpleEdgeImg, segLines, verifiedBox2, this->segRects);

        }

        return true;

    }
    else
        return false;

}

void TextLocation::thresholdImg(const IplImage* grayImg, int dilateDeg) {

    cvSmooth(grayImg, m_smoothImg, CV_GAUSSIAN, ksmoothSize, ksmoothSize);

    getEdgeImg(m_smoothImg, m_simpleEdgeImg, dilateDeg);

    double globalTh = getThForGlobalThImg(m_smoothImg, khistoRatio);

    if (globalTh > 50) {

        getGlobalThImg(m_smoothImg, globalTh, m_globalThImg, dilateDeg);

        cvAnd(m_globalThImg, m_simpleEdgeImg, m_advancedEdgeImg);
        cvAnd(m_advancedEdgeImg, blackPixelImg, m_edgeAndBlackImg);

        thrImgMode = ADVANCED_EDGE; // globalAndEdge
#ifdef DEBUG
        cvShowImage("advancedEdgeImg", m_advancedEdgeImg);
#endif

    } else {

        cvAnd(m_simpleEdgeImg, blackPixelImg, m_edgeAndBlackImg);

        thrImgMode = SIMPLE_EDGE; // edgeImg

    }
#ifdef DEBUG
    cvShowImage("edgeAndBlackImg", m_edgeAndBlackImg);
#endif

}

bool TextLocation::getBoundingBox(const IplImage* src, int &clusterNum, CvRect &box) {

    vector<CvRect> rects(0);
    vector<CvPoint> centers(0);

    maxNumLimitedConnectComponet(src, kboxMaxNum, rects, centers);

    removeSmallRect(rects, centers);
    removePillarRect(rects, centers);

    if (centers.size() == 0) {

        return false;

    } else {

        // no clustering
        clusterNum = 1;
        getUnitedRects(rects, box);

        return true;
    }



}

void TextLocation::getMaskImgFromRects(const IplImage* src, const vector<CvRect> &rects, IplImage* dst) {

    // use rects to mask src image (creat mask)

    cvZero(m_utilityMaskImg);

    for (unsigned int i = 0; i < rects.size(); ++i) {

        for (int row = rects[i].y; row < rects[i].y + rects[i].height; ++row) {

            uchar* pMask = (uchar*) (m_utilityMaskImg->imageData + row * m_utilityMaskImg->widthStep);

            for (int col = rects[i].x; col < rects[i].x + rects[i].width; ++col) {

                pMask[col] = 255;

            }
        }
    }


    cvZero(m_utilityZeroImg);
    cvOr(src, m_utilityZeroImg, dst, m_utilityMaskImg);

}

double TextLocation::getThForGlobalThImg(const IplImage* src, double ratio) {

    // 'hist': image pixel value histogram
    vector<int> hist(256, 0);
    for (int row = 0; row < src->height; row++) {
        uchar* pSrc = (uchar*) (src->imageData + row * src->widthStep);
        for (int col = 0; col < src->width; col++) {
            hist[pSrc[col]] += 1;
        }
    }

    // The accumulated pixel count from index 255 to 'idx' equals 'pixelNumTh'
    double acc = 0;
    double pixelNumTh = ratio * (src->width * src->height);

    int idx = 255;
    while (acc < pixelNumTh) {
        acc += hist[idx];
        idx--;
    }

    idx += kglobalThBias;

    return idx;


}

void TextLocation::maxNumLimitedConnectComponet(const IplImage* img, const int maxNum,
        vector<CvRect> &rects, vector<CvPoint> &centers) {

    cvCopy(img, m_copyForCC);
    cvDilate(m_copyForCC, m_copyForCC, NULL, 2);

    // connected component
    int ccNum = 200;
    connectComponent(m_copyForCC, 1, kccPerimeter, &ccNum, rects, centers);

    // too many ccNum, dilate the cc img then cc again
    cvZero(m_maskForCC);
    cvCopy(img, m_copyForCC);

    while (ccNum > kboxMaxNum) {

        getMaskImgFromRects(m_copyForCC, rects, m_maskForCC);
        cvDilate(m_maskForCC, m_maskForCC, NULL, 2);

#ifdef DEBUG
        cvShowImage("maskImg", m_maskForCC);
#endif

        cvCopy(m_maskForCC, m_copyForCC);
        connectComponent(m_maskForCC, 1, kccPerimeter, &ccNum, rects, centers);

    }

}

void TextLocation::removePillarRect(vector<CvRect> &rects, vector<CvPoint> &centers) {

    vector<CvRect>::iterator it = rects.begin();
    vector<CvPoint>::iterator it2 = centers.begin();

    while (it != rects.end()) {

        int rectWidth = it->width;
        int rectHeight = it->height;

        if ((double)rectHeight > 2.5*(double)rectWidth) {

            rects.erase(it);
            centers.erase(it2);

            it = rects.begin();
            it2 = centers.begin();

        } else {

            it++;
            it2++;
        }
    }

}

void TextLocation::getSegLines(const IplImage* img, CvRect numBox, vector<int> &segCenter) {

    // get the pixel value histogram of numBox rect in img
    vector<int> hist;
    getHorHistByRect(img, numBox, hist);

    // clearify the valley
    clearifyHist(hist, 10);

    plot1DHisto(hist, -1, "valley hist");

    getHistValleyCenters(hist, segCenter);

}

void TextLocation::getSegRects(const IplImage* src, const vector<int> segLines, CvRect box, vector<CvRect> &segRects) {

    cvCopy(src, forRecogImg);

    for (unsigned int i = 0; i <  (segLines.size() - 1); ++i) {

        CvRect rect = cvRect(box.x + segLines[i], box.y,
                segLines[i + 1] - segLines[i], box.height);

        segRects.push_back(rect);

    }

}

void TextLocation::removeSmallRect(vector<CvRect> &rects, vector<CvPoint> &centers) {

    int num = rects.size();

    if (num != 0) {

        vector<int> area;
        area.resize(num);
        for (int i = 0; i < num; ++i) {
            area[i] = rects[i].width * rects[i].height;
        }

        int sum_area = 0;
        for (int i = 0; i < num; ++i) {
            sum_area += area[i];
        }
        int avg_area = sum_area / num;

        vector<CvRect>::iterator it = rects.begin();
        vector<CvPoint>::iterator it2 = centers.begin();

        while (it != rects.end()) {

            int rectWidth = it->width;
            int rectHeight = it->height;

            if (rectWidth * rectHeight < (avg_area / kboxAreaThCoeff)) {

                rects.erase(it);
                centers.erase(it2);

                it = rects.begin();
                it2 = centers.begin();

            } else {

                it++;
                it2++;
            }
        }

    }
}

void TextLocation::getEdgeImg(const IplImage* src, IplImage* edgeImg, int dilateDeg) {

    cvCanny(src, edgeImg, 60, 150, 3);
    cvDilate(edgeImg, edgeImg, NULL, dilateDeg);
#ifdef DEBUG
    cvShowImage("edge", edgeImg);
#endif

}

void TextLocation::getGlobalThImg(const IplImage* src, double th, IplImage* globalThImg, int dilateDeg) {

    cvThreshold(src, globalThImg, th, 255, 0);
    inverseBinaryImage(m_globalThImg);
    cvDilate(m_globalThImg, m_globalThImg, NULL, dilateDeg);

}

void TextLocation::getUnitedRects(const vector<CvRect> &rects, CvRect &numberBox) {

    int leftMost = rects[0].x;
    int rightMost = rects[0].x + rects[0].width;
    int topMost = rects[0].y;
    int buttomMost = rects[0].y + rects[0].height;

    for (unsigned int i = 0; i < rects.size(); ++i) {

        if (rects[i].x < leftMost) {
            leftMost = rects[i].x;
        }
        if ((rects[i].x + rects[i].width) > rightMost) {
            rightMost = rects[i].x + rects[i].width;
        }
        if (rects[i].y < topMost) {
            topMost = rects[i].y;
        }
        if ((rects[i].y + rects[i].height) > buttomMost) {
            buttomMost = rects[i].y + rects[i].height;
        }
    }

    // add margin to bounding box
    leftMost = ((leftMost - 5) >= 0) ? (leftMost - 5) : leftMost;
    rightMost = ((rightMost + 5) < m_edgeAndBlackImg->width) ? (rightMost + 5) : rightMost;

    numberBox.x = leftMost;
    numberBox.y = topMost;
    numberBox.width = rightMost - leftMost;
    numberBox.height = buttomMost - topMost;

}

void TextLocation::getHistValleyCenters(const vector<int> &hist, vector<int> &centers) {

    int startIdx = 0;
    int endIdx = 0;
    centers.clear();

    for (unsigned int i = 1; i < hist.size(); ++i) {

        if ( (hist[i-1] != 0) && (hist[i] == 0) ) { // 1 -> 0
            startIdx = i;

        } else if ( (hist[i-1] == 0) && (hist[i] != 0) ) { // 0 -> 1

            endIdx = i;

            int idx = (startIdx + endIdx) / 2;
            centers.push_back(idx);

            startIdx = 0;
            endIdx = 0;

        } else if (i == hist.size() - 1) {

            if (startIdx != 0) {

                endIdx = i;

                int idx = (startIdx + endIdx) / 2;
                centers.push_back(idx);

            }

        }
    }
}

void TextLocation::getHistZeroIntervals(const vector<int> &hist, vector<int> &centers, vector<int> &widths) {

    int startIdx = 0;
    int endIdx = 0;
    centers.clear();
    widths.clear();

    if (hist[0] == 0) {

        startIdx = 0;

        for (unsigned int i = 1; i < hist.size(); ++i) {

            if ((hist[i - 1] != 0) && (hist[i] == 0)) { // 1 -> 0

                startIdx = i;

            } else if ((hist[i - 1] == 0) && (hist[i] != 0)) { // 0 -> 1

                endIdx = i;

                int idx = (startIdx + endIdx) / 2;
                int width = (endIdx - startIdx);
                centers.push_back(idx);
                widths.push_back(width);

                startIdx = 0;
                endIdx = 0;

            } else if (i == hist.size() - 1) {

                if (startIdx != 0) {

                    endIdx = i;

                    int idx = (startIdx + endIdx) / 2;
                    int width = (endIdx - startIdx);
                    centers.push_back(idx);
                    widths.push_back(width);

                }

            }
        }

    }
    else if (hist[0] != 0) {

        for (unsigned int i = 1; i < hist.size(); ++i) {

            if ((hist[i - 1] != 0) && (hist[i] == 0)) { // 1 -> 0

                startIdx = i;

            } else if ((hist[i - 1] == 0) && (hist[i] != 0)) { // 0 -> 1

                endIdx = i;

                int idx = (startIdx + endIdx) / 2;
                int width = (endIdx - startIdx);
                centers.push_back(idx);
                widths.push_back(width);

                startIdx = 0;
                endIdx = 0;

            } else if (i == hist.size() - 1) {

                if (startIdx != 0) {

                    endIdx = i;

                    int idx = (startIdx + endIdx) / 2;
                    int width = (endIdx - startIdx);
                    centers.push_back(idx);
                    widths.push_back(width);

                }

            }
        }

    }

}

TextLocation::~TextLocation() {

    cvReleaseImage(&grayImg);
    cvReleaseImage(&blackPixelImg);
    cvReleaseImage(&outImg);
    cvReleaseImage(&forRecogImg);

    cvReleaseImage(&m_edgeAndBlackImg);
    cvReleaseImage(&m_simpleEdgeImg);
    cvReleaseImage(&m_globalThImg);
    cvReleaseImage(&m_advancedEdgeImg);
    cvReleaseImage(&m_smoothImg);
    cvReleaseImage(&m_copyForCC);
    cvReleaseImage(&m_maskForCC);
    cvReleaseImage(&m_utilityMaskImg);
    cvReleaseImage(&m_utilityZeroImg);

}
