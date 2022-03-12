# Self Driving Project - AIDL
### Students: Helena Martin, Hermela Seleshi and Cecilia Siliato
### Advisor: Mariona Caròs

# Table of contents
1.  [Motivation](#Motivation)
2.  [Dataset](#Dataset)
3.  [Anlaysis of DataSet](#AnalysisofTheDataset)
4.  [Yolo V1 Architectures](#YoloV1Architectures)
5.  [Loss Function](#LossFunction)
6.  [Evaluation Matrix](#EvaluationMatrix)
7.  [Intersection Over Union](#IntersectionOverUnion)
8.  [Mean Avergae Precision](#MeanAveragePrecision)
9.  [compututional Resource](#CompututionalResource)
10. [Training Yolo V1](#TrainingYoloV1)
11. [Challenges](#Challenges)
12. [Exploiding Gradient](#ExploidingGradient)
13. [Transfer Learning](#TransferLearning)
14. [Introduction To Transfer Learning](#IntroductionToTransferLearning)
15. [Application Of Transfer Learning In This Project](#ApplicationOfTransferLearningInThisProjcet)
16. [Models Comparision](#ModelsComparision)
17. [Validation With Our Own Images](#ValidationWithOurOwnImages)
18. [Inference](#Inference)
19. [Conclusion And Future Work](#ConclusionAndFutureWork)
20. [Refernces](#Refernces)

## 1. Motivation

 **Problems and Challenges**

**The benefits of Autonomous vehicles**

* Greater road safety, reduce crash, and congestion,

* Independence for people with disabilities (blind, old)
  
**New potential risks**.  

Decision errors that may result in death and injury
Cybersecurity:


**Critical Tasks to increase benefits and mitigate risks**

Object detection, drivable area segmentation, lane detection, deep detection, etc

**Main Project Goal**

Focus on traffic  object detection (classification and localization)

Object detection is a crucial for object tracking, trajectory estimation,  and collision avoidanc

**Main Project Challenge**

Detect dynamic road elements (pedestrians, cyclist, vehicles) that are 
continuously changing location and behaviour under diverse lighting and background conditions

## Dataset
The Berkeley Deep Drive dataset contains a variety annotated images for 2D and 3D object detection, instance segmentation, lane markings, etc. For our project, we use the annotated images for 2D object detection.
The dataset consists over 100.000 video clips of driving videos in different conditions. For 2D object detection, an extraction on 100.000 clips is done to obtain images and the annotations of the bounding boxes.
The images are in RGB and have a size of 1280x720 pixels.
The annotations are provided in a JSON file including:
- Bounding Boxes and the corresponding object class
- Weather
- Time of the day
- Scene
 The video clips from where the images are extracted are filmed in different parts of the USA.
 **INSERT IMAGE LOCATIONS DATASET**
 
 ### Analysis of the Datset
 
 | ![alt text](https://github.com/hemahecodes/AIDL_SelfDrivingProject/blob/main/data/weather_plot_val.png?raw=true) | ![alt text](https://github.com/hemahecodes/AIDL_SelfDrivingProject/blob/main/data/weather_plot.png?raw=true) |
|:--:|:--:|
| *Number of images in each weather condition in validation data set* | *Number of images in each weather condition in test data set* |


 | ![alt text](https://github.com/hemahecodes/AIDL_SelfDrivingProject/blob/main/data/scene_plot_val.png?raw=true) | ![alt text](https://github.com/hemahecodes/AIDL_SelfDrivingProject/blob/main/data/scene_plot.png?raw=true) |
|:--:|:--:|
| *Number of images in each scene in validation data set* | *Number of images in each scene in test data set* | 
 
 | ![alt text](https://github.com/hemahecodes/AIDL_SelfDrivingProject/blob/main/data/timeofday_plot_val.png?raw=true)  | ![alt text](https://github.com/hemahecodes/AIDL_SelfDrivingProject/blob/main/data/timeofday_plot.png?raw=true) |
|:--:|:--:|
| *Number of images in each time of the day in validation data set* | *Number of images in each time of the day in test data set* |  
 
 
 
## YOLO v1: Architecture

You Only Look Once (YOLO) is an object detection model. The name is due to the fact that this algorithm is able to detect and recognize various objects in a picture (in real-time). It is also important to mention that nowadays there exists many versions of this model (v1, v2, v3, v4,...). We have selected the first one because of resources issues but also we considered really important having clear the main idea of this model and for so, it is enough working with YOLO v1.

The main idea of YOLO is first dividing the input image in a fixed number SxS cells as we can see on the image (in this example S=7):
![alt text](https://user-images.githubusercontent.com/94481725/156919394-0a670c9b-4c32-4f21-b4da-f84793e38d99.jpg)

After that, each of the SxS cells of the grid will be responsible to detect a maximum of 1 object of the image. It is important to know that we say that a cell is responsible for detecting an object if the center of the bounding box of this object is on the cell.

On the following example, the cell (4,3) would be the responsible for detecting the bike:

![alt text](https://user-images.githubusercontent.com/94481725/156919554-b71cf241-c44f-4c2a-a214-f4bb080f30e9.jpg)

In order to do this, YOLO v1 has an architecture consisting of 6 blocks combining convolutional layers with maxpooling layers and followed by 2 fully connected layers. Furthermore it applies the Leaky ReLu activation function after all layers except for the last one and uses dropout between the two fully connected layers in order to tackle overfitting.

![alt text](https://user-images.githubusercontent.com/94481725/156921245-b489fc5f-b218-41b8-9c38-27ca6a868e7b.jpg)

The dimension of the last fully connected layer would be SxSx(B * 5 + C) where:

* S: Number of cells in which input image is divided
* B: Number of bounding boxes that will predict each cell (normally B=2, so each cell is responsible to detect 2 bounding boxes and then 1 is discarded)
* C: Number of classes of our dataset

So, for each cell of the image you have:
* 2 bounding boxes predicted: each bounding box has 4 numbers (center coordinates and width/height) and the number remaining is the probability of having an object inside this bounding box (P(Object)
* Class probabilities: For each of our dataset classes, we have the conditional probability (P(class 1 | Object)... P(class C | Object))

It is more clear showed in this image:

![alt text](https://user-images.githubusercontent.com/94481725/156921416-21bb7fe4-35cc-48a5-878b-0a5cffa70b77.jpg)


So, we can then understand that when we execute the model, we will obtain SxSx2 bounding boxes but then it will be compared with the ground truth bounding boxes in order to keep only the ones with the highest IoU (intersection over union).

### Loss functions

The loss function in YOLO v1 is not one of the classic losses in neural networks. In this case, the loss is divided in different losses that we will see now:
**1. Bounding box loss**: This loss as the name suggests refers to the bounding box and it is divided into 2 different losses:

1.1. *Bounding box centroid loss*: This will be the distance of the center of the predicted bounding box and the ground truth bounding box but it is important to keep in mind that it is only computed when there actually exists an object on the studied cell. It is computed as follows: 

![alt text](https://user-images.githubusercontent.com/94481725/156934299-24356708-fede-4f9d-a460-eebf373cfcfc.jpg)

1.2. *Bounding box size loss*: This one is computed as the distance of the width and height of the predicted bounding box and the ground truth bounding box but it is important to keep in mind that it is only computed when there actually exists an object on the studied cell. In this case we have to keep in mind also that it is computed the sqrt because otherwise larger bounding boxes would have more importance on this loss.

The formal formula is:


![alt text](https://user-images.githubusercontent.com/94481725/156934449-92bfa605-1b96-4941-ad34-99e76fc6b261.jpg)

**2. Object loss**: This loss refers to the error that is done when assigning object probability and in the ground-truth there is an object. In other words, if there is an object on a particular cell, which is the difference between the P(Object) and 1? It is computed as follows:


![alt text](https://user-images.githubusercontent.com/94481725/156935053-01381f1d-f6a0-4d12-a2c7-4b13d3d37aa1.jpg)

**3. No Object loss**: This loss refers to the error that is done when assigning object probability and in the ground-truth there is not any object. In other words, if there is not any object on a particular cell, which is the difference between the P(Object) and 0? It is computed as follows:
![alt text](https://user-images.githubusercontent.com/94481725/156924120-642363df-40cd-4245-b736-a21a5c3c70d9.jpg)

**4. Class loss**: In this last loss, we are computing the error made when assigning a class to a detected object, so it is pretty similar as the previous loss but in this case, we are looking at the P(Class i | Object). The formula is:

![alt text](https://user-images.githubusercontent.com/94481725/156924363-1f7bb066-d6ef-4a7f-ae6d-b73be7d1e70c.jpg)

So, finally, if we add all these losses, we will obtain the loss of YOLO v1:

![alt text](https://user-images.githubusercontent.com/94481725/156935209-2b71f713-9d3c-4772-9613-3c9c88e92f16.jpg)

## Evaluation Metrics

We used several metrics to evaluate our object detection model.

### Intersection over Union (IoU)
Intersection over Union is a metric used to measure the overlap between two vounding voxes.
If the prediction is correct, the Iou is equal to 1. Therefore, the lower the IoU, the worse the prediction result.

| ![alt text](https://github.com/hemahecodes/AIDL_SelfDrivingProject/blob/main/data/IoU.png?raw=true) |
|:--:|
| *Computation of Intersection over Union* |


In our project we used the IoU to classify the predictions as True Positives (TP), False Positives (FP) and False Negatives (FN):
- IoU >= 0.5: The prediction is classified as a True Positive (TP).
- IoU <  0.5: The prediction is classified as a False Positive (FP).
- When the model failed to detect and object in an image, the prediction is classified as a False Negative (FN)

To evaluate the performance of the model, we used the previous tP, FP and FN classifications to compute the precision and recall of the model.
On the one hande, the **precision** measures how accurate are our predictions, thus, the percentage of predictions that are correct. On the other hand, **recall** measures how good the model finds all the positives in a set of predictions.

| ![alt text](https://github.com/hemahecodes/AIDL_SelfDrivingProject/blob/main/data/precision_recall.png?raw=true) |
|:--:|
| *Computation of Precision and Recall* |

### Mean Average Precision (mAP)
Average precision computes the average precision values for recall value oer 0 to 1. 
To compute the Average Precision we find the area under the curve of the precision-recall curve. 

## Computational Resources

**ADD NVIDIA-SMI OUTPUT AND EXLPAIN NUMBER OF CPUS, TYPE OF GPU ETC.**


## Training YOLO v1

## Challenges
### Exploding Gradients
At first, we experienced exploiding gradients while training our network.

Exploding Gradients are a problem where large error gradients accumulate. This happened because our model was unstable and unacapable of learning from the training data.
| ![alt text](https://github.com/hemahecodes/AIDL_SelfDrivingProject/blob/main/data/Exploding_gradients.png?raw=true) |
|:--:|
| *Example plot of exploding gradients* |

We observed this problem in it's extreme form, since the weight values resulted in NaN and can no longer be updated.

| ![alt text](https://github.com/hemahecodes/AIDL_SelfDrivingProject/blob/main/data/Exploding_gradients_nan.png?raw=true) |
|:--:|
| *Example plot of exploding gradients until NaN values* |


## Transfer Learning

### Introduction to Transfer Learning
First of all, we will make a very quick introduction to Transfer Learning. The main idea of TL is using models already trained on our custom dataset. In order to do that, there are some steps that should be done:

1. Select a model that performs the same (or a very similar) task as we want to do in order to take advantage of the features that the pretrained model already trained
2. Select the backbone that we want for our model (it can be useful for faster predictions, for example)
3. Load the parameters (weights and biases) from the pretrained model: In the case of pytorch, we have some models already pretrained for object detection (for instance Faster R-CNN and RetinaNet)
4. Finetune the model to better adapt to our dataset: Finetuning the model means changing some parameters (or even some layers) so then we can train it again with our dataset. In our particular case, we needed to adapt the pretrained model to a different number of classes (as the models were previously pretrained with COCO (91 classes) and we have less classes)
5. Train the model again for a few epochs: The idea no is training the model again but with our own data. Normally it is not needed to do it with the whole dataset but with a fewer subset (the main features are already learnt with the pretrained model and we only want to adapt it to our dataset)
6. The "retrained" model will now do some better predictions on our dataset

The main advantages of doing transfer learning are the saving of computer and time resources but also the "no-need" to have huge datasets that normally are difficult/expensive to label.

### Application of Transfer Learning in this project

As we have used the Pytorch environment on this project, we have taken advantage of some models already pretrained on pytorch and COCO dataset. So, we have selected 2 of the best performing models:

* Faster R-CNN: This is the last algorithm of the trilogy (R-CNN, Fast R-CNN and Faster R-CNN) and the main idea is that there are 2 subnetworks (2-stage object detector):
  
  1. Region Proposal Network: This network will be the responsible of purposing different regions in which may exist an object
  
  2. Classifier: Here, the object classification will be done once the RPN has send to it some region proposals

![alt text](https://user-images.githubusercontent.com/94481725/156935475-d38f8f50-90e9-482b-99c5-34bf1f1b7588.jpg)


* RetinaNet: This is a one-stage object detection model that utilizes a focal loss function to address class imbalance during training.

![alt text](https://user-images.githubusercontent.com/94481725/156935669-17567676-0f0e-4033-ac00-cc54477dc0e5.jpg)

The Transfer Learning codes are on the directory *transfer_learning* and in order to reproduce them, we have a mini subset of our dataset in *transfer_learning/data*. We can do the first part of "retraining the models" using the codes *train_FasterRCNN.py* and *train_RetinaNet.py*. 

On the first one, 3 arguments are needed: 

1. "b": It refers to the backbone used. Possibilities:

  1.1 b=1: MobileNetV3: Constructs a high resolution Faster R-CNN model with a MobileNetV3-Large FPN backbone
  
  1.2 b=2: ResNet50: Constructs a Faster R-CNN model with a ResNet-50-FPN backbone.
  
  1.3 b=3: MobileNetV3-320: Constructs a low resolution Faster R-CNN model with a MobileNetV3-Large FPN backbone tunned for mobile use-cases.
  
2. "c": It refers to the number of classes used (on the data subset we have 11 classes)
 
3. "e": Number of epochs for training the model

So, the main steps would be:

```
git clone https://github.com/hemahecodes/AIDL_SelfDrivingProject
cd transfer_learning
python train_FasterRCNN.py -b 1 -c 11 -e 20
```

This last instructions will train and save the model with the trained weights in *transfer_learning/models*, in the ".pth" file it will be specified the pretrained model and backbone used.

On the case of RetinaNet, we do not have to specify the backbone (as the only available is ResNet50) so we only should give the number of classes and epochs.

In order to do inference with the "retrained" models, we can use the *inference....py* scripts as follows:

```
python inference_FasterRCNN_resnet50.py -i "data/DeepDriving/test/fd5bae34-d63db3d7.jpg"
```


The image with the bounding boxes of the objects detected will be automatically showed and saved on *predictions/prediction_FastRCNN-ResNet50fd5bae34-d63db3d7.jpg*. In this particular case, the result would be:

![alt text](https://user-images.githubusercontent.com/94481725/156932373-8892b364-6e24-4b0f-b2ab-65c55c0e201b.jpg)

So, we can see that it has worked pretty well. In general, we have seen the best results when using FastRCNN with ResNet50 backbone and the worst with Fast R-CNN and MobileNet v3-320. In fact, if we compare this image predicted with the other pretrained models:

**FastRCNN with MobileNet v3**

![alt text](https://user-images.githubusercontent.com/94481725/156932786-ebf34c90-201e-4dfd-8faf-3e70467aeb49.jpg)

**FastRCNN with MobileNet v3-320**

![alt text](https://user-images.githubusercontent.com/94481725/156933141-8235ec4c-92e9-445c-bfa9-9ee19f29a083.jpg)

**FastRCNN with ResNet50**

![alt text](https://user-images.githubusercontent.com/94481725/156933380-d23eb36c-e577-4f0c-a8cc-8ea7df6ab430.jpg)

**RetinaNet with ResNet50**

![alt text](https://user-images.githubusercontent.com/94481725/156934024-869e4bc5-58d8-4c49-b34d-696e32e5c25b.jpg)



## Models comparison

## Validation with our own images

## Inference

## Conclusion and future work
 In our project we implemented and trained a one stage detector YOLO and two stage detector Faster R-CNN on the BDD 100K dataset in the context of of autonomous vehicles.

 The main result: we just explore one of the main critical tasks, namely object detection.

 Since we used the first version of YOLO in this project, in the future we must experiment with newer models, for instance the most recent  versions of YOLO,

 In the Future we will need to explore these three tasks simultaneously. We may add a four critical task: deep detection. This would allow us to reach performance with high accuracy and high FPS which are suitable for the goal of autonomous driving.

 Since autonomous vehicles take decisions involving matters of life and death, we will have to find new ways to train the autonomous vehicle to make these complex decisions.

 **Generally we would need higher precision and real-time driving perception system that can assist the autonomous vehicle in making the reasonable decision while driving safely.**


## 9. References

Gene Lewis. Object detection for autonomous vehicles, 2014.

 Jason Brownlee. A gentle introduction to object recognition with deep learning. Machine Learning Mastery, 5, 2019.

S. Ren, K. He, R. Girshick, and J. Sun, “Faster r-cnn: Towards realtime object detection with region proposal networks,” arXiv preprint arXiv:1506.01497, 2015

 A. Bochkovskiy, C.-Y. Wang, and H.-Y. M. Liao, “Yolov4: Optimal speed and accuracy of object detection,” arXiv preprint arXiv:2004.10934, 2020

 A. Paszke, A. Chaurasia, S. Kim, and E. Culurciello, “Enet: A deep neural network architecture for real-time semantic segmentation,” arXiv preprint arXiv:1606.02147, 2016

H. Zhao, J. Shi, X. Qi, X. Wang, and J. Jia, “Pyramid scene parsing network,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), July 2017

 X. Pan, J. Shi, P. Luo, X. Wang, and X. Tang, “Spatial as deep: Spatial cnn for traffic scene understanding,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 32, no. 1, 2018

Y. Hou, Z. Ma, C. Liu, and C. C. Loy, “Learning lightweight lane detection cnns by self-attention distillation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019, pp. 1013–1021

 C.-Y. Wang, A. Bochkovskiy, and H.-Y. M. Liao, “Scaled-yolov4: Scaling cross stage partial network,” arXiv preprint arXiv:2011.08036, 2020.

Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. You only look once: Unified, real-time object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 779-788, 2016.



