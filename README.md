# SVHN-YOLOV3-CUSTOM

# Github
The full Github link would be in here: https://github.com/StephenEkaputra/SVHN-YOLOV3-CUSTOM

We divided the code into 3 main folders: Data Preprocessing, Training, and Testing.

Weight can be downloaded: https://drive.google.com/file/d/1K7tI4n95-yd4wDS_0NAtvMWqo3TPL1LR/view?usp=sharing

# Computer
Ubuntu 16.04 LTS
i7-9700K CPU @3.60Ghz x 8
RTX 2060
OS 64 bit

# Data Preprocessing
We collected SVHN dataset containing 33402 for training and 13068 for testing. Then, we extracted the annotation digitStruct.mat into normal data annotations.

After that, we converted the normal data annotations to Yolo annotation format using code from: http://guanghan.info/blog/en/my-works/train-yolo/

But, however the code was wrong because it can produce negative value and we realized that the number of the annotation would be from 0 to 1. So, we fixed the formula after we read the Yolo paper. 

[class x_center y_center width height]

These annotations should be in the same folder with train images / test images.

# Training
We trained YoloV3 from: https://github.com/AlexeyAB/darknet

The first thing to do is we make the file (setting the GPU=1, CUDNN=1). We used CUDA v10.1 and CuDNN to train faster.

After that, we calculated the anchors by using command: 
./darknet detector calc_anchors data/obj.data -num_of_clusters 9 -width 512 -height 256 -show

Then, we got the anchor values and replaced the anchors from yolo-obj.cfg. These anchor values were just a matter of size that fit well to be trained.

Generated anchor values:

29,67, 37,112, 57,112, 46,166, 64,150, 66,195, 85,166, 90,205, 121,204

Further, we set the parameter of yolo-obj.cfg. We followed AlexeyAB’s instructions in parameter setting and we modified them.
 
The class in obj.names is from 0 to 10 (since number 0 = 10) not 1 to 10 because yolo was only able to train from index 0. So, the class would be 11 not 10.

Furthermore, we trained the data by using this command:
./darknet detector train data/obj.data data/yolo-obj.cfg data/darknet53.conv.74 -mjpeg_port 8090

In this part, we used pretrained weight darknet53.conv.74 to get better result. For the model architecture, we used YoloV3.

# Testing
In this section, we tested some of the weights obtained from training process against 13,068 images. We found that yolo-obj_best.weights was the best result. Then, we saved the predicted list to json file. We got the highest mAP score of 0.41474.
 
# Speed Benchmark
We tried to test the weight on Google Colab and we got the result 1.07 s per loop.
 
# Experiments
We had tried to train yolo with Google Colab but it stopped at 2000th iteration due to the usage limitation. So, we trained yolo with a computer (GPU powered) and stopped at 22000th iteration. The last accuracy was 76.99%.
 
We also had tried to set the width and height size parameter in yolo-obj.cfg to 416x416 respectively but the result is not good enough considering the original size of training images were not square. Then, we changed it to 512x256 and get a good result. Those size has to be able to be divided by 32 and we realized that the scale of the training images were 2:1. Moreover, we also tried to set the flip parameter to 0 because no digit numbers are reversed.

Further, we had tried to change the image angle, saturation, exposure, and hue but the result is not quite good. The reason is because the image is bright enough to be trained and we didn’t see any differences in angle variation of the number that were too significant. 

In addition, we used batch size of 64 and subdivision of 32 due to GPU memory. If the batch size is too large or the subdivision is too small, then out of GPU memory would be happened.

# Conclusion
To train yolo, we need to figure out the data size (width and height) and the data annotation coordinates should be in range of 0 to 1. In addition, we don’t recommend to train yolo with huge dataset using Google Colab despite using free GPU because it costs a lot of time to synchronize the data and it will stop training at 2000th iteration. We also don’t recommend to use data augmentation because it could make the result worse except we set the flip to 0. In the end, there would be a trade-off between accuracy and speed.
