#include <opencv2/opencv.hpp>
#include "logger.hpp"
using namespace cv;

double FindBestMatchRect(Mat &fullImage, Mat &templateImage, Rect &outRect)
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

double FindBestMatchRect(Mat &fullImage, Mat &templateImage, Rect &outRect, Mat mask)
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
Mat TrimImage(Mat &source)
{
    Mat gray, bin;
    vector<vector<Point>> contours;

    // グレースケールからの二値化(背景は真っ黒想定なのでthreshは１)
    cvtColor(source, gray, COLOR_BGR2GRAY);
    threshold(gray, bin, 1, 255, THRESH_BINARY);

    // 輪郭検出
    vector<Vec4i> hierarchy;
    findContours(bin, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

    // 輪郭のおさまるRectを検出し、切り取り。
    //(背景は真っ黒想定なので、index=0の輪郭が問題なく画像の輪郭であると言える。)
    Rect bound = boundingRect(contours[0]);
    Mat trimmed(source, bound);
    return trimmed;
}

// 特徴点を抽出し、ホモグラフィー行列を計算し、templateImageに適用して返す。
Mat HomograpyTransformIMG(Mat &fullImage, Mat &templateImage, double DISTANCE_THRESH, DescriptorMatcher::MatcherType matcherType)
{
    auto detector = AKAZE::create(AKAZE::DESCRIPTOR_MLDB, 0, 3, 0.001f);

    vector<KeyPoint> templateKeypoints, sourceKeypoints;
    Mat templateDiscriptors, sourceDescriptors;
    detector->detectAndCompute(templateImage, noArray(), templateKeypoints, templateDiscriptors);
    detector->detectAndCompute(fullImage, noArray(), sourceKeypoints, sourceDescriptors);

    Ptr<DescriptorMatcher> matcher = BFMatcher::create(matcherType, true);

    // 最も近い点のみでのmatch行列を取得
    vector<DMatch> matches;
    matcher->match(templateDiscriptors, sourceDescriptors, matches);

    /*
    sort(matches.begin(), matches.end(), [](auto const &left, auto const &right)
         { return left.distance < right.distance; });

    vector<DMatch> matches;
    for (int i = 0; i < allMatches.size(); i++)
    {
        // distanceが一つ前と比較してデカすぎたら(distanceが急にデカくなる=異物の可能性が高い)
        if (i != 0 && allMatches[i - 1].distance < allMatches[i].distance * DISTANCE_THRESH)
        {
            break;
        }
        else
        {
            matches.push_back(allMatches[i]);
        }
    }
    */

    // マッチした特徴点のペアを取得
    vector<Point2f> matchTemplateKeypoints, matchSourceKeypoints;
    for (size_t i = 0; i < matches.size(); i++)
    {
        matchTemplateKeypoints.push_back(templateKeypoints[matches[i].queryIdx].pt);
        matchSourceKeypoints.push_back(sourceKeypoints[matches[i].trainIdx].pt);
    }

    Mat masks;
    Mat homography = findHomography(matchTemplateKeypoints, matchSourceKeypoints, masks, RANSAC, 3);

    /*
    Mat matchImage;
    drawMatches(templateImage, templateKeypoints, fullImage, sourceKeypoints, matches, matchImage);
    imshow("matchImage", matchImage);
    */

    Mat transformedImg;
    warpPerspective(templateImage, transformedImg, homography, fullImage.size());

    // imshow("warped", transformedImg);

    // 画像サイズでトリム
    transformedImg = TrimImage(transformedImg);

    // imshow("trimmed", transformedImg);

    return transformedImg;
}

void SampleAlphaMask(Mat &inImage, Mat &outMask, int threshold)
{
    int width = inImage.rows;
    int height = inImage.cols;
    Mat gray;
    Mat binary;

    cvtColor(inImage, gray, COLOR_BGR2GRAY);
    cv::threshold(gray, outMask, threshold, 255, THRESH_BINARY);
}

void logDenoizeResult(String params[], logger &log)
{
    String p;
    for (int i = 0; i < 5; i++)
    {
        p = params[i];
        log.append(p);
    }
    log.nextLine();
}

int main()
{
    logger LOGGER("log.csv", "src,method,threshold,score,error,");
    const string FULL_IMAGE_PATH = "sample.jpeg";
    const double MATCH_THRESHOLD = 0.8;
    const int BINARY_THRESHOLD = 100;

    const Vec2i correctCords(885, 505);
    std::vector<String> templatePathList;
    std::string noizePath = "noize/";
    std::filesystem::directory_iterator iter(noizePath), end;
    std::error_code err;

    for (; iter != end && !err; iter.increment(err))
    {
        const filesystem::directory_entry entry = *iter;
        const string path = entry.path().string();
        if (path.find(".jpeg") != std::string::npos)
        {
            templatePathList.push_back(path);
        }
    }

    for (auto itr = templatePathList.begin(); itr != templatePathList.end(); ++itr)
    {
        String logParams[5];
        Mat templateImage = imread(*itr);
        Mat fullImage = imread(FULL_IMAGE_PATH);
        Rect outRect;

        // denoiseなし
        logParams[0] = *itr;
        logParams[1] = "NONE";
        logParams[2] = "NONE";
        logParams[3] = to_string(FindBestMatchRect(fullImage, templateImage, outRect));
        logParams[4] = "\"(" + to_string(correctCords[0] - outRect.x) + "," + to_string(correctCords[1] - outRect.y) + ")\"";
        logDenoizeResult(logParams, LOGGER);

        int count = 0;
        Mat mask;
        String method;
        double score;
        bool endFlag = false;

        // denoiseセクション
        while (atof(logParams[3].c_str()) < MATCH_THRESHOLD)
        {

            switch (count)
            {
            case 0:
            {
                /*
                 暗部除去:
                 明るい背景映像に対して、陰になっている部分や、
                 天井が見切れている部分などを除去。
                */
                SampleAlphaMask(templateImage, mask, BINARY_THRESHOLD);
                score = FindBestMatchRect(fullImage, templateImage, outRect, mask);
                method = "BINAR";
                break;
            }
                /*
                case 1:
                    method = "FLANNBASEDFD";
                    FindBestMatchWithFD(fullImage, templateImage, outRect, mask, DescriptorMatcher::FLANNBASED);

                    break;
                    */

            case 1:
            {
                method = "BRUTEFORCEFD_FD";

                // 上で生成したmaskを使いつつ、特徴量検出によるホモグラフィー変換(+位置補正)をかける。
                Mat transformedImg = HomograpyTransformIMG(fullImage, templateImage, 0.75, DescriptorMatcher::BRUTEFORCE_HAMMING);
                SampleAlphaMask(transformedImg, mask, BINARY_THRESHOLD);
                score = FindBestMatchRect(fullImage, transformedImg, outRect, mask);
                break;
            }
            default:
                endFlag = true;
                break;
            }

            if (endFlag)
            {
                break;
            }
            logParams[1] = method;
            logParams[2] = to_string(BINARY_THRESHOLD);
            logParams[3] = to_string(score);
            logParams[4] = "\"(" + to_string(correctCords[0] - outRect.x) + "," + to_string(correctCords[1] - outRect.y) + ")\"";
            logDenoizeResult(logParams, LOGGER);

            if (score > MATCH_THRESHOLD)
            {
                break;
            }
            count++;
        }

        // ログを残す
        logDenoizeResult(logParams, LOGGER);
    }

    // 終了時にログを書き込む
    LOGGER.writeFile();
    /*
    rectangle(fullImage, outRect, Scalar(0, 255, 255), 3);
    imshow("result", fullImage);
    */
    waitKey();
}