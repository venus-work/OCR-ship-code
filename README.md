<div align="center">

# OCR Ship's Code - Nanodet & Paddle OCR | OCR 
A comprehensive object detection & OCR system both Nanodet (Supper fast and high accuracy lightweight) object detection and PaddleOCR (Multiple text-type recognition and Handwriting recognition), specifically designed for public spaces on lake.
</div>

## TODO
- Video / Image 


### Cách sử dụng

#### 1. Chạy bằng dòng lệnh
* Inference image
```bash
python demo2.py image --config CONFIG_PATH  --model MODEL_PATH --path IMAGE_PATH --save_result
```
* Inference video

```bash
python demo2.py video --config CONFIG_PATH  --model MODEL_PATH --path VIDEO_PATH --save_result
```
* Inference webcam

```bash
python demo/demo.py webcam --config CONFIG_PATH --model MODEL_PATH --camid YOUR_CAMERA_ID

```
* CONFIG_PATH: 'path of yml file model'
* MODEL_PATH: 'path of ckpt file'
* IMAGE_PATH: 'path of image'
* VIDEO_PATH: 'path of video'
* --save_result: save .txt file
****

## Install

### Requirements

* Linux or MacOS
* CUDA >= 10.2
* Python >= 3.7
* Pytorch >= 1.10.0, <2.0.0

### Step

1. **Clone the repository**
```bash
git clone <respository-url>
cd nanodet
```

2. **Create virtual environment**
```bash
python -m venv venv
#On Windows
venv\Scripts\activate
#On Linux/Mac
source venv/bin/activate
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```
6. **Download model weights**

## Model Zoo

NanoDet supports variety of backbones. Go to the [***config*** folder](config/) to see the sample training config files.
NanoDet supports variety of backbones. Go to the [***config*** folder](config/) to see the sample training config files.

Model                 | Backbone           |Resolution|COCO mAP| FLOPS |Params | Pre-train weight |
:--------------------:|:------------------:|:--------:|:------:|:-----:|:-----:|:-----:|
NanoDet-m             | ShuffleNetV2 1.0x  | 320*320  |  20.6  | 0.72G | 0.95M | [Download](https://drive.google.com/file/d/1ZkYucuLusJrCb_i63Lid0kYyyLvEiGN3/view?usp=sharing) |
NanoDet-Plus-m-320 (***NEW***)     | ShuffleNetV2 1.0x | 320*320  |  27.0  | 0.9G  | 1.17M | [Weight](https://drive.google.com/file/d/1Dq0cTIdJDUhQxJe45z6rWncbZmOyh1Tv/view?usp=sharing) &#124; [Checkpoint](https://drive.google.com/file/d/1YvuEhahlgqxIhJu7bsL-fhaqubKcCWQc/view?usp=sharing)
NanoDet-Plus-m-416 (***NEW***)     | ShuffleNetV2 1.0x | 416*416  |  30.4  | 1.52G | 1.17M | [Weight](https://drive.google.com/file/d/1FN3WK3FLjBm7oCqiwUcD3m3MjfqxuzXe/view?usp=sharing) &#124; [Checkpoint](https://drive.google.com/file/d/1gFjyrl7O8p5APr1ZOtWEm3tQNN35zi_W/view?usp=sharing)
NanoDet-Plus-m-1.5x-320 (***NEW***)| ShuffleNetV2 1.5x | 320*320  |  29.9  | 1.75G | 2.44M | [Weight](https://drive.google.com/file/d/1Xdlgu5lxiS3w6ER7GE1mZpY663wmpcyY/view?usp=sharing) &#124; [Checkpoint](https://drive.google.com/file/d/1qXR6t3TBMXlz6GlTU3fxiLA-eueYoGrW/view?usp=sharing)
NanoDet-Plus-m-1.5x-416 (***NEW***)| ShuffleNetV2 1.5x | 416*416  |  34.1  | 2.97G | 2.44M | [Weight](https://drive.google.com/file/d/16FJJJgUt5VrSKG7RM_ImdKKzhJ-Mu45I/view?usp=sharing) &#124; [Checkpoint](https://drive.google.com/file/d/17sdAUydlEXCrHMsxlDPLj5cGb-8-mmY6/view?usp=sharing)


*Notice*: The difference between `Weight` and `Checkpoint` is the weight only provide params in inference time, but the checkpoint contains training time params.


Legacy Model Zoo

Model                 | Backbone           |Resolution|COCO mAP| FLOPS |Params | Pre-train weight |
:--------------------:|:------------------:|:--------:|:------:|:-----:|:-----:|:-----:|
NanoDet-m-416         | ShuffleNetV2 1.0x  | 416*416  |  23.5  |  1.2G | 0.95M | [Download](https://drive.google.com/file/d/1jY-Um2VDDEhuVhluP9lE70rG83eXQYhV/view?usp=sharing)|
NanoDet-m-1.5x        | ShuffleNetV2 1.5x  | 320*320  |  23.5  | 1.44G | 2.08M | [Download](https://drive.google.com/file/d/1_n1cAWy622LV8wbUnXImtcvcUVPOhYrW/view?usp=sharing) |
NanoDet-m-1.5x-416    | ShuffleNetV2 1.5x  | 416*416  |  26.8  | 2.42G | 2.08M | [Download](https://drive.google.com/file/d/1CCYgwX3LWfN7hukcomhEhGWN-qcC3Tv4/view?usp=sharing)|
NanoDet-m-0.5x        | ShuffleNetV2 0.5x  | 320*320  |  13.5  |  0.3G | 0.28M | [Download](https://drive.google.com/file/d/1rMHkD30jacjRpslmQja5jls86xd0YssR/view?usp=sharing) |
NanoDet-t             | ShuffleNetV2 1.0x  | 320*320  |  21.7  | 0.96G | 1.36M | [Download](https://drive.google.com/file/d/1TqRGZeOKVCb98ehTaE0gJEuND6jxwaqN/view?usp=sharing) |
NanoDet-g             | Custom CSP Net     | 416*416  |  22.9  |  4.2G | 3.81M | [Download](https://drive.google.com/file/d/1f2lH7Ae1AY04g20zTZY7JS_dKKP37hvE/view?usp=sharing)|
NanoDet-EfficientLite | EfficientNet-Lite0 | 320*320  |  24.7  | 1.72G | 3.11M | [Download](https://drive.google.com/file/d/1Dj1nBFc78GHDI9Wn8b3X4MTiIV2el8qP/view?usp=sharing)|
NanoDet-EfficientLite | EfficientNet-Lite1 | 416*416  |  30.3  | 4.06G | 4.01M | [Download](https://drive.google.com/file/d/1ernkb_XhnKMPdCBBtUEdwxIIBF6UVnXq/view?usp=sharing) |
NanoDet-EfficientLite | EfficientNet-Lite2 | 512*512  |  32.6  | 7.12G | 4.71M | [Download](https://drive.google.com/file/d/11V20AxXe6bTHyw3aMkgsZVzLOB31seoc/view?usp=sharing) |
NanoDet-RepVGG        | RepVGG-A0          | 416*416  |  27.8  | 11.3G | 6.75M | [Download](https://drive.google.com/file/d/1nWZZ1qXb1HuIXwPSYpEyFHHqX05GaFer/view?usp=sharing) |



*Notice*: The difference between `Weight` and `Checkpoint` is the weight only provide params in inference time, but the checkpoint contains training time params.


****

## Thanks

https://github.com/RangiLyu/nanodet

https://github.com/PaddlePaddle/PaddleOCR
