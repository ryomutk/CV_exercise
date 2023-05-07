#include "Denoize.hpp"
using namespace OpencvWrapper;

double Denoize::AutoDenoize(Mat &fullImage, Mat &templateImage, std::string &denoizeMethods, Rect &outRect, bool saveResults, bool useTemplateMatching)
{
    int n = int(DenoizeType::NUM_ITEMS);
    std::vector<DenoizeType> denoizes;
    Mat outMat;
    Mat mask(templateImage.size(), templateImage.type());
    double score;

    // ビット全探索の要領で、全ての組み合わせでデノイズができるようにする
    // ノイズタイプを特定できるものが欲しいが、通常そこまで長くはならない。
    for (int bit = 0; bit < (1 << n); ++bit)
    {
        denoizes.clear();
        for (int i = 0; i < n; ++i)
        {
            if (bit & (1 << i))
            {
                denoizes.push_back(DenoizeType(i));
            }
        }

        mask.setTo(Scalar(255, 255, 255));
        outMat = templateImage.clone();
        denoizeMethods = "";

        for (auto itr = denoizes.begin(); itr != denoizes.end(); ++itr)
        {
            switch (*itr)
            {
            case DenoizeType::MEAN_DENOISING:
                denoizeMethods += "MEAN_DENOISING:";
                MeansDenoising(outMat, outMat);
                /*
                imshow("denoized",outMat);
                waitKey();
                */
                break;
            case DenoizeType::BINARIZATION_MASK:
                SampleAlphaMask(outMat, mask, BINARY_THRESHOLD);
                denoizeMethods += "BINARIZATION_MASK:";
                imwrite("denoize_results/mask.jpeg", mask);
                break;

            default:
                break;
            }
        }

        if (saveResults)
        {
            imwrite("denoize_results/" + denoizeMethods + ".jpeg", outMat);
        }

        if (!useTemplateMatching)
        {
            //imshow("in", outMat);
            bool result = HomograpyTransformAndMatchCoord(fullImage, outMat, outMat, mask, outRect, DescriptorMatcher::BRUTEFORCE_HAMMING, 0.4);

            if (result)
            {
                std::cout << "successed with " << denoizeMethods << std::endl;
                return score;
            }
            else
            {
                std::cout << "failed with " << denoizeMethods << std::endl;
            }
        }
        else
        {
            HomograpyTransformAndMatchCoord(fullImage, outMat, outMat, mask, outRect, DescriptorMatcher::BRUTEFORCE_HAMMING, 0.4);

            if (std::find(denoizes.begin(), denoizes.end(), DenoizeType::BINARIZATION_MASK) != denoizes.end())
            {
                SampleAlphaMask(outMat, mask, BINARY_THRESHOLD);
            }
            else
            {
                mask = Mat(outMat.size(), outMat.type(), Scalar(255, 255, 255));
            }

            score = TemplateMatch(fullImage, outMat, outRect, mask);

            /*
            imshow("mask", mask);
            imshow("in", templateImage);
            imshow("out", outMat);
            Mat resultImg = fullImage.clone();
            rectangle(resultImg, outRect, Scalar(0, 255, 0), 4);
            imshow("result", resultImg);
            */

            if (score > MATCH_THRESHOLD)
            {
                return score;
            }
        }
    }

    return 0;
}

double Denoize::TemplateMatch(Mat &fullImage, Mat &templateImage, Rect &outRect)
{
    Mat resized_template, resized_full, resized_mask;
    double RESIZE_RATE;
    if (templateImage.rows > templateImage.cols)
    {
        RESIZE_RATE = std::min(RESIZE_BASE / templateImage.cols, 1.0);
    }
    else
    {
        RESIZE_RATE = std::min(RESIZE_BASE / templateImage.rows, 1.0);
    }

    resize(templateImage, resized_template, Size(), RESIZE_RATE, RESIZE_RATE);
    resize(fullImage, resized_full, Size(), RESIZE_RATE, RESIZE_RATE);

    Mat result;

    matchTemplate(resized_full, resized_template, result, TM_CCOEFF_NORMED);

    outRect.height = resized_template.cols;
    outRect.width = resized_template.rows;
    Point maxPt;
    double maxValue;

    minMaxLoc(result, NULL, &maxValue, NULL, &maxPt);

    std::cout << "(" << maxPt.x << "," << maxPt.y << ")"
              << "score:" << maxValue << "\n";

    outRect.x = maxPt.x / RESIZE_RATE;
    outRect.y = maxPt.y / RESIZE_RATE;
    outRect.width /= RESIZE_RATE;
    outRect.height /= RESIZE_RATE;

    return maxValue;
}

double Denoize::TemplateMatch(Mat &fullImage, Mat &templateImage, Rect &outRect, Mat &mask)
{
    Mat resized_template, resized_full, resized_mask;
    double RESIZE_RATE;
    if (templateImage.rows > templateImage.cols)
    {
        RESIZE_RATE = std::min(RESIZE_BASE / templateImage.cols, 1.0);
    }
    else
    {
        RESIZE_RATE = std::min(RESIZE_BASE / templateImage.rows, 1.0);
    }
    resize(templateImage, resized_template, Size(), RESIZE_RATE, RESIZE_RATE);
    resize(fullImage, resized_full, Size(), RESIZE_RATE, RESIZE_RATE);
    resize(mask, resized_mask, Size(), RESIZE_RATE, RESIZE_RATE);

    Mat result;

    matchTemplate(resized_full, resized_template, result, TM_CCOEFF_NORMED, resized_mask);
    outRect.height = resized_template.cols;
    outRect.width = resized_template.rows;
    Point maxPt;
    double maxValue;

    minMaxLoc(result, NULL, &maxValue, NULL, &maxPt);

    std::cout << "(" << maxPt.x << "," << maxPt.y << ")"
              << "score:" << maxValue << "\n";

    outRect.x = maxPt.x / RESIZE_RATE;
    outRect.y = maxPt.y / RESIZE_RATE;
    outRect.width /= RESIZE_RATE;
    outRect.height /= RESIZE_RATE;

    return maxValue;
}

// Homography変換後の、マッチ部分の座標を求める
void Denoize::CalcRect(Mat &source, Rect &outRect)
{
    Mat gray, bin;
    std::vector<std::vector<Point>> contours;

    int mx(std::numeric_limits<int>::max()), my(std::numeric_limits<int>::max()), Mx(0), My(0);

    for (int x = 0; x < source.cols; x++)
    {
        for (int y = 0; y < source.rows; y++)
        {
            if (source.at<Vec3b>(y, x) != Vec3b::zeros())
            {
                mx = min(x, mx);
                my = min(y, my);
                Mx = max(x, Mx);
                My = max(y, My);
            }
        }
    }

    /*
    // グレースケールからの二値化(背景は真っ黒想定なのでthreshは１)
    cvtColor(source, gray, COLOR_BGR2GRAY);
    threshold(gray, bin, 1, 255, THRESH_BINARY);

    // 輪郭検出
    std::vector<Vec4i> hierarchy;
    findContours(bin, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    */

    // 輪郭のおさまるRectを検出し、切り取り。
    //(背景は真っ黒想定なので、index=0の輪郭が問題なく画像の輪郭であると言える。)
    // outRect = boundingRect(contours[0]);
    outRect = Rect(Point(mx, my), Point(Mx, My));
}

// 特徴点を抽出し、ホモグラフィー行列を計算し、templateImageに適用して返す。
bool Denoize::HomograpyTransformAndMatchCoord(Mat &fullImage, Mat &templateImage, Mat &outImage, Mat &mask, Rect &resultRect, DescriptorMatcher::MatcherType matcherType, double DISTANCE_THRESH)
{
    auto detector = AKAZE::create(AKAZE::DESCRIPTOR_MLDB, 0, 3, 0.001f);

    Mat grayFull, grayTemp;
    cvtColor(fullImage, grayFull, COLOR_RGB2GRAY);
    cvtColor(templateImage, grayTemp, COLOR_RGB2GRAY);

    std::vector<KeyPoint> templateKeypoints, sourceKeypoints;
    Mat templateDiscriptors, sourceDescriptors;
    detector->detectAndCompute(grayTemp, mask, templateKeypoints, templateDiscriptors);
    detector->detectAndCompute(grayFull, noArray(), sourceKeypoints, sourceDescriptors);

    Ptr<DescriptorMatcher> matcher = BFMatcher::create(matcherType);

    // 指定された数のknnでmatchを行う
    std::vector<std::vector<DMatch>> knnMatches;
    matcher->knnMatch(templateDiscriptors, sourceDescriptors, knnMatches, 2);

    std::vector<DMatch> matches;
    for (int i = 0; i < knnMatches.size(); i++)
    {
        // matches.push_back(knnMatches[i][0]);

        // 1番と2番の差が極端＝無関係なもの（ノイズ）な可能性が高い
        if (knnMatches[i][0].distance < knnMatches[i][1].distance * DISTANCE_THRESH)
        {
            matches.push_back(knnMatches[i][0]);
        }
    }

    // マッチした特徴点のペアを取得
    std::vector<Point2f> matchTemplateKeypoints, matchSourceKeypoints;
    for (size_t i = 0; i < matches.size(); i++)
    {
        matchTemplateKeypoints.push_back(templateKeypoints[matches[i].queryIdx].pt);
        matchSourceKeypoints.push_back(sourceKeypoints[matches[i].trainIdx].pt);
    }

    /*
    Mat matchImage;

    drawMatches(templateImage, templateKeypoints, fullImage, sourceKeypoints, matches, matchImage);
    imshow("matchImage", matchImage);
    waitKey();
    */

    if (matches.size() >= 4)
    {
        Mat masks;
        Mat homography = findHomography(matchTemplateKeypoints, matchSourceKeypoints, masks, RANSAC, 3);

        if (homography.cols == 0)
        {
            outImage = templateImage.clone();
            return false;
        }

        cv::warpPerspective(templateImage, outImage, homography, fullImage.size());

        // imshow("warped", outImage);

        // 座標情報を返す
        CalcRect(outImage, resultRect);

        /*
        Mat resultIm = outImage.clone();
        circle(resultIm, Point(resultRect.x + resultRect.width / 2, resultRect.y + resultRect.height / 2), 10, Scalar(255, 0, 0));
        rectangle(resultIm, resultRect, Scalar(0, 255, 0), 3);
        imshow("resultim", resultIm);
        waitKey();
        */

        // 画像サイズでトリム
        outImage = Mat(outImage, resultRect);

        // imshow("trimmed", outImage);
        // waitKey();
        return true;
    }
    else
    {
        return false;
    }
}

void Denoize::SampleAlphaMask(Mat &inImage, Mat &outMask, int threshold)
{
    int width = inImage.rows;
    int height = inImage.cols;
    Mat gray;
    Mat binary;

    cvtColor(inImage, gray, COLOR_BGR2GRAY);
    cv::threshold(gray, outMask, threshold, 255, THRESH_BINARY);
}

void Denoize::MeansDenoising(Mat &inImage, Mat &outImage, double strength)
{
    fastNlMeansDenoisingColored(inImage, outImage, 10, 10, 7, 21);
}
