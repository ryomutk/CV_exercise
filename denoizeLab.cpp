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

//特徴点を抽出し、ホモグラフィー行列を計算し、templateImageに適用して返す。
Mat HomograpyTransformIMG(Mat &fullImage, Mat &templateImage, Rect &outRect, Mat mask, DescriptorMatcher::MatcherType matcherType)
{
    auto detector = AKAZE::create(AKAZE::DESCRIPTOR_MLDB, 0, 3, 0.001f);

    vector<KeyPoint> templateKeypoints, sourceKeypoints;
    Mat templateDiscriptors, sourceDescriptors;
    detector->detectAndCompute(templateImage, noArray(), templateKeypoints, templateDiscriptors);
    detector->detectAndCompute(fullImage, noArray(), sourceKeypoints, sourceDescriptors);

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(matcherType);

    // 最も近い点のみでのmatch行列を取得
    vector<vector<DMatch>> knnMatches;
    matcher->knnMatch(templateDiscriptors, sourceDescriptors, knnMatches, 20);

    vector<DMatch> matches;
    const float ratio_thresh = .75f;
    for (size_t i = 0; i < knnMatches.size(); i++)
    {
        if (knnMatches[i][0].distance <
            ratio_thresh * knnMatches[i][1].distance)
        {
            matches.push_back(knnMatches[i][0]);
        }
    }

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
                method = "BRUTEFORCEFD";

                Mat transformedImg = HomograpyTransformIMG(fullImage, templateImage, outRect, mask, DescriptorMatcher::BRUTEFORCE_HAMMING);
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

        logDenoizeResult(logParams, LOGGER);
    }

    LOGGER.writeFile();
    /*
    rectangle(fullImage, outRect, Scalar(0, 255, 255), 3);
    imshow("result", fullImage);
    */
    waitKey();
}
