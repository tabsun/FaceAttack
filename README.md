# Face Attack

Based on gradient of face recognition ensemble.

This code gets best score @ 4.31 which rank 10th in [Ali's competition of Anti-FaceRecognition](https://tianchi.aliyun.com/competition/entrance/231745/introduction). 

## Requirements

tensorflow

opencv-python

...

## Pre-trained models

You can download the needed model files from [Baidu Cloud](https://pan.baidu.com/s/1ViSYoOWCYOxZeX_9_7iUmQ) but I know the downloading speed will be a pity:(

## How to run it?

1. Generate all 712 images' embeddings by different networks. Here I used six models including MX-insightFace's models and TF-insightFace's models because InsightFace is well-known for its performance on LFW. But most models' backbones are resnet series, a better choice is to add some different backbones like mobilenet or inception.

```

python3 calculate_embedding.py

```

If you have any problem with this step, you can just forget it and continue to step 2. Because I have added my pre-calculated results under proper directory. You can find them under ./csvs.

2. Generate adversarial examples by adding gradient to the original image. This repeated operation will make the input image of networks closer to another face and further from original face at the same time. I used cosine-distance, so the loss is similar with triplet loss which is introduced by facenet.

```

python3 adv.py

```

After this, the code will save ad-images under ./adv_images with name like 00004.jpg_3.2345.jpg which denotes the delta's norm.

## Notes

1. I tried targeted method with arcface loss and the best score is almost 7.x. The generality seems a bottleneck.

2. If you have any questions or interests in this code, welcome to send a mail to buptmsg@gmail.com.

## Citations

[InsightFace_TF](https://github.com/auroua/InsightFace_TF)

[InsightFace_MXNet](https://github.com/deepinsight/insightface)

[Model conversion](https://github.com/microsoft/MMdnn/issues/135)

[FaceNet](https://github.com/davidsandberg/facenet)


