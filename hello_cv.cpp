#include <opencv2/opencv.hpp>
using namespace cv;

double FindBestMatchRect(Mat &fullImage, Mat &templateImage, Rect &outRect)
{
    Mat result;

    matchTemplate(fullImage, templateImage, result, TM_CCOEFF_NORMED);

    outRect.height = templateImage.cols;
    outRect.width = templateImage.rows;
    Point maxPt;
    double maxValue;

    minMaxLoc(result, NULL, &maxValue, NULL, &maxPt);

    std::cout << "(" << maxPt.x << "," << maxPt.y << ")"
              << "score:" << maxValue<<"\n";

    outRect.x = maxPt.x;
    outRect.y = maxPt.y;

    return maxValue;
}

double FindBestMatchRect(Mat &fullImage, Mat &templateImage, Rect &outRect, Mat mask)
{

    Mat result;

    matchTemplate(fullImage, templateImage, result, TM_CCOEFF_NORMED, mask);
    outRect.height = templateImage.cols;
    outRect.width = templateImage.rows;
    Point maxPt;
    double maxValue;

    minMaxLoc(result, NULL, &maxValue, NULL, &maxPt);

    std::cout << "(" << maxPt.x << "," << maxPt.y << ")"
              << "score:" << maxValue << "\n";

    outRect.x = maxPt.x;
    outRect.y = maxPt.y;

    return maxValue;
}

void SampleAlphaMask(Mat &inImage, Mat &outMask)
{
    int width = inImage.rows;
    int height = inImage.cols;
    Mat gray;
    Mat binary;

    cvtColor(inImage, gray, COLOR_BGR2GRAY);
    // カメラによってダイナミックレンジが違いそうなので、閾値を自動で設定するadaptiveSamplingを採用
    //adaptiveThreshold(gray, outMask, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 11, 2);
    // adaptiveThreshold(gray, outMask, 255, CALIB_CB_ADAPTIVE_THRESH, THRESH_BINARY, 11, 2);
    cv::threshold(gray,outMask,100,255,THRESH_BINARY);
}

int main()
{
    const double MATCH_THRESHOLD = 0.8;
    String srcPath = "noize1.jpeg";
    /*
    std::cout << "src?";
    std::cin >> srcPath;
    */
    Mat templateImage = imread(srcPath);
    Mat fullImage = imread("suzsiki.jpeg");
    Rect outRect;
    if (FindBestMatchRect(fullImage, templateImage, outRect) < MATCH_THRESHOLD)
    {
        Mat mask;
        // ここを動的に検出できるようにする
        SampleAlphaMask(templateImage, mask);

        //imshow("mask", mask);
        int max = FindBestMatchRect(fullImage, templateImage, outRect, mask);
    }

    rectangle(fullImage, outRect, Scalar(0, 255, 255), 3);

    imshow("result", fullImage);
    waitKey();
}
