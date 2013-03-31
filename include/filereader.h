
#ifndef _FILEREADER_H_
#define _FILEREADER_H_

#include <string>
#include <cv.h>
#include <highgui.h>
using namespace std;

class FileReader {
public:

    FileReader(int argc, const char** argv);

    IplImage* scaledImg;

    bool readFile(string filePath);
    string extractFilename(const string& path);

    ~FileReader();

private:

    const double kLongsideMax;

    IplImage* m_srcImg;

    void scaleImg(const IplImage &srcImg);

};


#endif
