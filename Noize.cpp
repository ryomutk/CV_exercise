#include "Noize.hpp"

using namespace OpencvWrapper;

void Noize::uturikomi(Mat &inputImage, Mat &outImage, std::string noizeMaskPath, float scale)
{
    Mat mask = imread(noizeMaskPath, IMREAD_UNCHANGED);

    double RESIZE_RATE;
    if (inputImage.rows > inputImage.cols)
    {
        RESIZE_RATE = (double)inputImage.rows / mask.rows;
    }
    else
    {
        RESIZE_RATE = (double)inputImage.cols / mask.cols;
    }
    resize(mask, mask, Size(), RESIZE_RATE * scale, RESIZE_RATE * scale);

    Point maskDelta((inputImage.cols - mask.cols) / 2, (inputImage.rows - mask.rows) / 2);

    outImage = inputImage.clone();
    Point maskPoint;

    for (int y = std::max(maskDelta.y, 0); y < std::max(maskDelta.y, 0) + std::min(inputImage.rows, mask.rows); y++)
    {
        for (int x = std::max(maskDelta.x, 0); x < std::max(maskDelta.x, 0) + std::min(inputImage.cols, mask.cols); x++)
        {
            maskPoint.y = y - maskDelta.y;
            maskPoint.x = x - maskDelta.x;
            if (mask.at<Vec4b>(maskPoint.y, maskPoint.x)[3] != 0)
            {
                outImage.at<Vec3b>(y, x)[0] = mask.at<Vec4b>(maskPoint.y, maskPoint.x)[0];
                outImage.at<Vec3b>(y, x)[1] = mask.at<Vec4b>(maskPoint.y, maskPoint.x)[1];
                outImage.at<Vec3b>(y, x)[2] = mask.at<Vec4b>(maskPoint.y, maskPoint.x)[2];
            }
        }
    }
}

void Noize::darken(Mat &inputImage, Mat &outImage, float rate)
{
    outImage = inputImage.clone();

    // グレーを乗算合成することで暗くすることにした。
    for (int i = 0; i < inputImage.rows; i++)
    {
        for (int j = 0; j < inputImage.cols; j++)
        {

            outImage.at<Vec3b>(i, j)[0] *= rate;
            outImage.at<Vec3b>(i, j)[1] *= rate;
            outImage.at<Vec3b>(i, j)[2] *= rate;
        }
    }
}

void Noize::deform(Mat &inputImage, Mat &outImage, double rotate, Vec2f scale)
{
    Point2f center(inputImage.cols / 2, inputImage.rows / 2);
    Mat affineMatrix = getRotationMatrix2D(center, rotate, 1);
    warpAffine(inputImage, outImage, affineMatrix, inputImage.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar::all(0));
    resize(outImage, outImage, Size(), scale[0], scale[1]);
}

void Noize::gaussianNoise(Mat &inputImage, Mat &outImage, float strength)
{
    Mat noise(inputImage.size(), inputImage.type());
    float stddev = 40;
    float mean = 128 * strength;
    cv::randn(noise, mean, stddev);
    outImage = inputImage + noise;
}
