#include <opencv2/opencv.hpp>
using namespace cv;
#define THRESHOLD 0.8;

double FindBestMatchRect(Mat &targetImage, Mat &templateImage, Rect &outRect)
{

    Mat result;
    matchTemplate(targetImage, templateImage, result, TM_CCOEFF_NORMED);
    outRect.height = targetImage.cols;
    outRect.width = targetImage.rows;
    Point maxPt;
    double maxValue;
    minMaxLoc(result, NULL, &maxValue, NULL, &maxPt);

    std::cout << "(" << maxPt.x << "," << maxPt.y << ")"
              << "score:" << maxValue;

    outRect.x = maxPt.x;
    outRect.y = maxPt.y;

    return maxValue;
}

int main()
{
    String srcPath = "part.jpeg";
    /*
    std::cout << "src?";
    std::cin >> srcPath;
    */
    Mat partImage = imread(srcPath);
    Mat templateImage = imread("suzsiki.jpeg");
    Rect outRect;
    FindBestMatchRect(partImage, templateImage, outRect);
    rectangle(templateImage, outRect, Scalar(0, 255, 255), 3);

    imshow("result", templateImage);
    waitKey();
}
