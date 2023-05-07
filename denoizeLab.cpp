#include <opencv2/opencv.hpp>
#include "Denoize.hpp"
#include "Noize.hpp"
#include "logger.hpp"

using namespace OpencvWrapper;

void logDenoizeResult(String params[], logger &log)
{
    String p;
    for (int i = 0; i < 5; i++)
    {
        p = params[i];
        cout << p << endl;
        log.append(p);
    }
    log.nextLine();
}

int main()
{
    logger LOGGER("log.csv", "name,noize,method,score,error");
    const string FULL_IMAGE_PATH = "full.jpeg";
    Mat fullImage = imread(FULL_IMAGE_PATH, COLOR_RGB2GRAY);

    std::vector<String> samplePathList;
    std::string samplePath = "samples/";
    std::filesystem::directory_iterator iter(samplePath), end;
    std::error_code err;
    
    //ディレクトリからサンプル画像を読み込み
    for (; iter != end && !err; iter.increment(err))
    {
        const filesystem::directory_entry entry = *iter;
        const string path = entry.path().string();
        if (path.find(".jpeg") != std::string::npos || path.find(".png") != std::string::npos)
        {
            samplePathList.push_back(path);
        }
    }

    //テスト用画像の繰り返し単位
    for (auto sampleItr = samplePathList.begin(); sampleItr != samplePathList.end(); ++sampleItr)
    {

        String logParams[5];
        Mat sampleImage = imread(*sampleItr, COLOR_RGB2GRAY);

        logParams[0] = *sampleItr;

        Rect outRect;
        double score;
        bool endFlag = false;
        Mat noizedImage;

        Denoize::TemplateMatch(fullImage, sampleImage, outRect);
        const Vec2i correctCords(outRect.x, outRect.y);

        //ノイズをかけるセクション
        for (int i = 0; i < int(NoizeTypes::NUM_ITEMS); i++)
        {
            NoizeTypes noize = NoizeTypes(i);
            
            //弱いノイズから始めて、どこまでデノイズできるかを確かめる
            for (float level = 0.4f; level < 2; level += 0.2f)
            {
                switch (noize)
                {
                case NoizeTypes::FLAT_UTURIKOMI:
                    Noize::uturikomi(sampleImage, noizedImage, "masks/mask.png", level);
                    logParams[1] = "FLAT_UTURIKOMI";
                    break;
                case NoizeTypes::COMPLEX_UTURIKOMI:
                    Noize::uturikomi(sampleImage, noizedImage, "masks/complex_mask.PNG", level);
                    logParams[1] = "COMPLEX_UTURIKOMI";
                    break;
                case NoizeTypes::DARKEN:
                    Noize::darken(sampleImage, noizedImage, level);
                    logParams[1] = "DARKEN";
                    break;
                case NoizeTypes::ROTATE:
                    Noize::deform(sampleImage, noizedImage, 180 * level);
                    imshow("deformed", noizedImage);
                    logParams[1] = "ROTATE";
                    break;
                case NoizeTypes::STRETCH:
                    Noize::deform(sampleImage, noizedImage, 180 * level, Vec2f(0.4 / level, 1));
                    imshow("deformed", noizedImage);
                    logParams[1] = "STRECH";
                    break;
                case NoizeTypes::SQUEEZE:
                    Noize::deform(sampleImage, noizedImage, 180 * level, Vec2f(level / 0.4, 1));
                    imshow("deformed", noizedImage);
                    logParams[1] = "SQUEEZE";
                    break;
                case NoizeTypes::GAUSSIAN_NOISE:
                    Noize::gaussianNoise(sampleImage, noizedImage, 1 * level);
                    logParams[1] = "GAUSSIAN_NOISE";
                    break;
                case NoizeTypes::NUM_ITEMS:
                    break;
                }

                logParams[1] += "(" + to_string(level) + ")";

                string denoizeMethods;
                score = Denoize::AutoDenoize(fullImage, noizedImage, denoizeMethods, outRect);
                
                //ログを残す
                if (score != 0)
                {
                    logParams[2] = denoizeMethods;
                    logParams[3] = to_string(score);
                    logParams[4] = "\"(" + to_string(correctCords[0] - outRect.x) + "," + to_string(correctCords[1] - outRect.y) + ")\"";
                    logDenoizeResult(logParams, LOGGER);
                }
                else
                {
                    logParams[2] = "FAILED";
                    logParams[3] = "FAILED";
                    logParams[4] = "FAILED";
                    logDenoizeResult(logParams, LOGGER);
                    break;
                }
            }
        }
    }

    // 終了時にログを書き込む
    LOGGER.writeFile();
    /*
    rectangle(fullImage, outRect, Scalar(0, 255, 255), 3);
    imshow("result", fullImage);
    */
}
