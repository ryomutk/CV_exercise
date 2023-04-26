#include <opencv2/opencv.hpp>
using namespace cv;

double FindBestMatchRect(Mat &targetImage, Mat &templateImage, Rect &outRect, Mat *mask = NULL)
{

    Mat result;
    matchTemplate(targetImage, templateImage, result, TM_CCOEFF_NORMED);
    outRect.height = targetImage.cols;
    outRect.width = targetImage.rows;
    Point maxPt;
    double maxValue;
    if (mask == NULL)
    {
        minMaxLoc(result, NULL, &maxValue, NULL, &maxPt);
    }
    else
    {
        minMaxLoc(result, NULL, &maxValue, NULL, &maxPt, *mask);
    }

    std::cout << "(" << maxPt.x << "," << maxPt.y << ")"
              << "score:" << maxValue;

    outRect.x = maxPt.x;
    outRect.y = maxPt.y;

    return maxValue;
}

int main()
{
    const double MATCH_THRESHOLD = 0.8;
    const double BINARIZATION_THRESHOLD = 240;
    String srcPath = "noize1.jpeg";
    /*
    std::cout << "src?";
    std::cin >> srcPath;
    */
    Mat partImage = imread(srcPath);
    Mat templateImage = imread("suzsiki.jpeg");
    Rect outRect;
    if (FindBestMatchRect(partImage, templateImage, outRect) < MATCH_THRESHOLD)
    {
        // ここを動的に検出できるようにする
        Mat mask = imread("mask.png");
        imshow("mask",mask);

        //int max = FindBestMatchRect(partImage, templateImage, outRect, &mask);
        //std::cout << "max:" << max;
    }

    rectangle(templateImage, outRect, Scalar(0, 255, 255), 3);

    imshow("result", templateImage);
    waitKey();
}
