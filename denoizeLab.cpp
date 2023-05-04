#include <opencv2/opencv.hpp>
#include "Denoize.hpp"
#include "logger.hpp"

using namespace OpencvWrapper;

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
    logger LOGGER("log.csv", "src,method,score,error");
    const string FULL_IMAGE_PATH = "full.jpeg";
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
        String logParams[4];
        Mat templateImage = imread(*itr);
        Mat fullImage = imread(FULL_IMAGE_PATH);
        Rect outRect;
        double score;
        bool endFlag = false;

        // denoiseセクション
        // 特徴量検出によるホモグラフィー変換(+位置補正)をかける。
        Mat transformedImg = Denoize::HomograpyTransformIMG(fullImage, templateImage, 0.9, DescriptorMatcher::BRUTEFORCE_HAMMING);
        Mat mask;
        // 黒背景をマスク
        Denoize::SampleAlphaMask(transformedImg, mask, 1);
        score = Denoize::FindBestMatchRect(fullImage, transformedImg, outRect, mask);

        // imshow("transformed",transformedImg);
        // imshow("fullimage",fullImage);

        logParams[0] = *itr;
        logParams[1] = "HOMOGRAPHY";
        logParams[2] = to_string(score);
        logParams[3] = "\"(" + to_string(correctCords[0] - outRect.x) + "," + to_string(correctCords[1] - outRect.y) + ")\"";

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
