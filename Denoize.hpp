#include <opencv2/opencv.hpp>

namespace OpencvWrapper
{
    using namespace cv;

    class Denoize
    {
    private:
        static Mat TrimImage(Mat &source);

    public:
        static double FindBestMatchRect(Mat &fullImage, Mat &templateImage, Rect &outRect);
        static double FindBestMatchRect(Mat &fullImage, Mat &templateImage, Rect &outRect, Mat mask);
        static Mat HomograpyTransformIMG(Mat &fullImage, Mat &templateImage, double DISTANCE_THRESH, DescriptorMatcher::MatcherType matcherType);
        static void SampleAlphaMask(Mat &inImage, Mat &outMask, int threshold);
    };
};