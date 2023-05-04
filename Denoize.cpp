#include "Denoize.hpp"
using namespace OpencvWrapper;

double Denoize::FindBestMatchRect(Mat &fullImage, Mat &templateImage, Rect &outRect)
{
    Mat result;

    matchTemplate(fullImage, templateImage, result, TM_CCOEFF_NORMED);

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

double Denoize::FindBestMatchRect(Mat &fullImage, Mat &templateImage, Rect &outRect, Mat mask)
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

// 画像の背景部分を切り取る。
Mat Denoize::TrimImage(Mat &source)
{
    Mat gray, bin;
    std::vector<std::vector<Point>> contours;

    // グレースケールからの二値化(背景は真っ黒想定なのでthreshは１)
    cvtColor(source, gray, COLOR_BGR2GRAY);
    threshold(gray, bin, 1, 255, THRESH_BINARY);

    // 輪郭検出
    std::vector<Vec4i> hierarchy;
    findContours(bin, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

    // 輪郭のおさまるRectを検出し、切り取り。
    //(背景は真っ黒想定なので、index=0の輪郭が問題なく画像の輪郭であると言える。)
    Rect bound = boundingRect(contours[0]);
    Mat trimmed(source, bound);
    return trimmed;
}

// 特徴点を抽出し、ホモグラフィー行列を計算し、templateImageに適用して返す。
Mat Denoize::HomograpyTransformIMG(Mat &fullImage, Mat &templateImage, double DISTANCE_THRESH, DescriptorMatcher::MatcherType matcherType)
{
    auto detector = AKAZE::create(AKAZE::DESCRIPTOR_MLDB, 0, 3, 0.001f);

    std::vector<KeyPoint> templateKeypoints, sourceKeypoints;
    Mat templateDiscriptors, sourceDescriptors;
    detector->detectAndCompute(templateImage, noArray(), templateKeypoints, templateDiscriptors);
    detector->detectAndCompute(fullImage, noArray(), sourceKeypoints, sourceDescriptors);

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

    Mat matchImage;
    drawMatches(templateImage, templateKeypoints, fullImage, sourceKeypoints, matches, matchImage);
    imshow("matchImage", matchImage);
    waitKey();

    Mat masks;
    Mat homography = findHomography(matchTemplateKeypoints, matchSourceKeypoints, masks, RANSAC, 3);

    Mat transformedImg;
    warpPerspective(templateImage, transformedImg, homography, fullImage.size());

    // imshow("warped", transformedImg);

    // 画像サイズでトリム
    transformedImg = TrimImage(transformedImg);

    // imshow("trimmed", transformedImg);

    return transformedImg;
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