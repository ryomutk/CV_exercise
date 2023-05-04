#include <opencv2/opencv.hpp>

namespace OpencvWrapper
{
    class Noize
    {
    public:
        static void uturikomi(cv::Mat inputImage, cv::Mat outImage, string noizeMaskPath, float scale = 1.0f);
        static void darken(cv::Mat inputImage, cv::Mat outImage, float rate = 1.0f);
        static void deform(cv::Mat inputImage, cv::Mat outImage, int rotate, cv::Vec2f scale = cv::Vec2f::ones());
    };
};
