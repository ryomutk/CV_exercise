#include <opencv2/opencv.hpp>

namespace OpencvWrapper
{
    using namespace cv;
    enum class NoizeTypes : int
    {
        FLAT_UTURIKOMI,
        COMPLEX_UTURIKOMI,
        DARKEN,
        ROTATE,
        STRETCH,
        SQUEEZE,
        GAUSSIAN_NOISE,
        NUM_ITEMS
    };

    class Noize
    {
    public:
        static void uturikomi(Mat &inputImage, Mat &outImage, std::string noizeMaskPath, float scale = 1.0f);
        static void darken(Mat &inputImage, Mat &outImage, float rate = 0.2f);
        static void deform(Mat &inputImage, Mat &outImage, double rotate, Vec2f scale = Vec2f::ones());
        static void gaussianNoise(Mat &inputImage, Mat &outImage, float strength = 1);
    };
};
