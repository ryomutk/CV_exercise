#include <opencv2/opencv.hpp>

namespace OpencvWrapper
{
    using namespace cv;
    class Noize
    {
    public:
        static void uturikomi(Mat inputImage, Mat outImage, std::string noizeMaskPath, float scale = 1.0f);
        static void darken(Mat inputImage, Mat outImage, float rate = 1.0f);
        static void deform(Mat inputImage, Mat outImage, int rotate, Vec2f scale = Vec2f::ones());
    };
};
