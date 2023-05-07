#include <opencv2/opencv.hpp>

namespace OpencvWrapper
{
    using namespace cv;
    enum class DenoizeType : int
    {
        MEAN_DENOISING,
        BINARIZATION_MASK,
        NUM_ITEMS
    };

    static const double MATCH_THRESHOLD = 0.8;
    static const int BINARY_THRESHOLD = 100;
    static const double RESIZE_BASE = 100;

    class Denoize
    {
    private:
        static void CalcRect(Mat &source, Rect &resultRect);

    public:
        static double AutoDenoize(Mat &fullImage, Mat &templateImage, std::string &denoizeMethods, Rect &outRect, bool saveResults = false, bool useTemplateMatching = false);
        static double TemplateMatch(Mat &fullImage, Mat &templateImage, Rect &outRect);
        static double TemplateMatch(Mat &fullImage, Mat &templateImage, Rect &outRect, Mat &mask);
        static bool HomograpyTransformAndMatchCoord(Mat &fullImage, Mat &templateImage, Mat &outImage, Mat &mask, Rect &resultRect, DescriptorMatcher::MatcherType matcherType = DescriptorMatcher::BRUTEFORCE_HAMMING, double DISTANCE_THRESH = 1);
        static void SampleAlphaMask(Mat &inImage, Mat &outMask, int threshold);
        static void MeansDenoising(Mat &inImage, Mat &outImage, double strength = 1.0);
    };
};
