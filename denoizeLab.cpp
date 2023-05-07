#include <opencv2/opencv.hpp>
#include <future>
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

void NoizeTest()
{
    logger LOGGER("log.csv", "name,noize,method,score,error");
    const string FULL_IMAGE_PATH = "full.jpeg";
    Mat fullImage = imread(FULL_IMAGE_PATH, COLOR_RGB2GRAY);

    std::vector<String> samplePathList;
    std::string samplePath = "samples/";
    std::filesystem::directory_iterator iter(samplePath), end;
    std::error_code err;

    // ディレクトリからサンプル画像を読み込み
    for (; iter != end && !err; iter.increment(err))
    {
        const filesystem::directory_entry entry = *iter;
        const string path = entry.path().string();
        if (path.find(".jpeg") != std::string::npos || path.find(".png") != std::string::npos)
        {
            samplePathList.push_back(path);
        }
    }

    // テスト用画像の繰り返し単位
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

        // ノイズをかけるセクション
        for (int i = 0; i < int(NoizeTypes::NUM_ITEMS); i++)
        {
            NoizeTypes noize = NoizeTypes(i);

            // 弱いノイズから始めて、どこまでデノイズできるかを確かめる
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
                    imwrite("noizedImages/gaussianNoise.png", noizedImage);
                    logParams[1] = "GAUSSIAN_NOISE";
                    break;
                case NoizeTypes::NUM_ITEMS:
                    break;
                }

                logParams[1] += "(" + to_string(level) + ")";

                string denoizeMethods;
                score = Denoize::AutoDenoize(fullImage, noizedImage, denoizeMethods, outRect, false, true);

                // ログを残す
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

void Roll(Mat &inImage, Mat &outImage, int &value)
{
    for (int i = 0; i < inImage.rows; i++)
    {
        for (int j = 0; j < inImage.cols; j++)
        {
            outImage.at<Vec3b>(i, j)[0] = inImage.at<Vec3b>(i, (j + value) % inImage.rows)[0];
            outImage.at<Vec3b>(i, j)[1] = inImage.at<Vec3b>(i, (j + value) % inImage.rows)[1];
            outImage.at<Vec3b>(i, j)[2] = inImage.at<Vec3b>(i, (j + value) % inImage.rows)[2];
        }
    }
}

// 画面におさまるように変形
void FitImage(Mat &screenImg, Mat &inImage, Mat &rescaledImage)
{
    double rateX, rateY, rate;
    rateX = double(screenImg.cols) / inImage.cols;
    rateY = double(screenImg.rows) / inImage.rows;
    rate = min({rateX, rateY, 1.0});

    resize(inImage, rescaledImage, Size(), rate, rate);
}

void CamDenoize()
{
    VideoCapture videoCapture(0);

    Mat vidImage, fullImage, denoizeImage;
    String input;
    Rect screenRect;

    int frameRate = 20;

    while (!videoCapture.isOpened())
    {
        videoCapture.open(0);
        waitKey(1000 / frameRate);
        cout << "waiting for camera";
    }

    videoCapture.read(vidImage);

    fullImage = imread("full.jpeg");
    screenRect.x = 0;
    screenRect.y = 0;
    screenRect.width = fullImage.cols / 2;
    screenRect.height = fullImage.rows - 1;

    int fourcc = VideoWriter::fourcc('m', 'p', '4', 'v');
    VideoWriter videoWriter("cam_rec.mp4", fourcc, frameRate, vidImage.size());

    Mat outImage(fullImage.size(), fullImage.type()), scrollImage(outImage, screenRect);

    int scrollVal = 0;
    bool gaugeanFlag = false;
    bool shootFlag = false;
    std::string denoizeMethods;
    Rect outRect;
    vector<Point> pointCords;

    logger LOGGER("video_log.csv", "name,method,score");
    while (true)
    {
        videoCapture.read(vidImage);
        Roll(fullImage, outImage, scrollVal);
        int key = waitKey(1000 / frameRate);

        if (key == 'g')
        {
            gaugeanFlag = !gaugeanFlag;
        }
        else if (key == 'e')
        {
            shootFlag = true;
        }
        else if (key == 'c')
        {
            pointCords.clear();
        }
        else if (key == 'q')
        {
            break;
        }

        if (gaugeanFlag)
        {
            Noize::gaussianNoise(outImage, outImage, 0.4);
        }

        if (shootFlag)
        {
            FitImage(scrollImage, vidImage, denoizeImage);
            Denoize::AutoDenoize(scrollImage, denoizeImage, denoizeMethods, outRect, true);
            Point newPoint = outRect.tl();
            newPoint.x += outRect.width / 2;
            newPoint.y += outRect.height / 2;
            pointCords.push_back(newPoint);
            shootFlag = false;
        }

        for (auto pItr = pointCords.begin(); pItr != pointCords.end(); ++pItr)
        {
            circle(scrollImage, *pItr, 10, Scalar(255, 0, 0), 5);
        }

        videoWriter.write(vidImage);
        imshow("video", scrollImage);
        imshow("cam", vidImage);
        setWindowProperty("video", WINDOW_KEEPRATIO, WINDOW_KEEPRATIO);
        scrollVal += 10;
    }

    videoWriter.release();
}

int main()
{
    std::string mode;
    std::cout << "Test?:y/n(n)";
    std::cin >> mode;
    if (mode == "y")
    {
        // サンプル画像でノイズテストし、ログを残す
        NoizeTest();
    }
    else
    {
        // カメラからの入力をdenoize
        CamDenoize();
    }
}
