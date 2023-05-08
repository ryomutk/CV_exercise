# Getting Started

## コンパイル方法
プロジェクトフォルダ内で以下のコマンドを実行してコンパイルしてください
```
g++ -fdiagnostics-color=always -g denoizeLab.cpp -o denoizeLab -std=c++17 `pkg-config --libs opencv4` `pkg-config --cflags opencv4` logger.cpp Noize.cpp Denoize.cpp
```
## 実行

実行時、
```Test?:y/n(n)```
と聞かれます。yを入力するとDennoiseTester,
それ以外だとCamDenoiserが実行されます

## お詫び
(この文章含め、"Denoise”と”Denoize”の表記揺れが多数あります。自分でも完全に認識が揺れてました。いつか直します。)
