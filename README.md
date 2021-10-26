# Detection of Personal-Protective-Equipment (PPE)
## Introduction
A PyTorch implementation of real-time detection of PPE (e.g., hard hat, safety vest) compliances of workers. 
## Problem
According from [the paper](https://www.sciencedirect.com/science/article/abs/pii/S0926580519308325) presents three different approaches for verifying PPE compliance:

<img src="https://github.com/ciber-lab/pictor-ppe/blob/master/extras/graphics/methods.jpg" align="middle"/>

**Approach-1**: Model detects worker, hat, and vest (three object classes) individually. Next, ML classifiers (Neural Network, Decision Tree) classify each worker as W (wearing no hat or vest), WH (wearing only hat), WV (wearing only vest), or WHV (wearing both hat and vest).

**Approach-2**: Model localizes workers in the input image and directly classifies each detected worker as W, WH, WV, or WHV.

**Approach-3**: Model first detects all workers in the input image and then, a CNN-based classifier model (VGG-16, ResNet-50, Xception) was applied to the cropped worker images to classify the detected worker as W, WH, WV, or WHV.

## Our implementation
This repository is implemented based-on the **Approach-1**, but instead of using YOLO model we use the end-to-end object detection method introduced in the article: [Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection](https://arxiv.org/abs/2006.04388).

### Overall impression
The article introduced Generalized Focal Loss improved from the original Focal Loss. A recent trend in one-stage detector is to introduce an individual prediction branch to estimate the quality of localization, e.g., the centerness FCOS or IoU score branch IoUNet can be trained separately and used in NMS process. They have the following problems in existing practices:

**Inconsistent usage of localization quality estimation and classification score between training and inference:** The localization quality estimation and classification score are usually trained independently but compositely utilized (e.g., multiplication) during inference; The supervision of the localization quality estimation is currently assigned for positive samples only, which is unreliable as negatives may get chances to have uncontrollably higher quality predictions. These two factors result in a gap between training and test, and would potentially degrade the detection performance, e.g., negative instances with randomly high-quality scores could rank in front of positive examples with lower quality prediction during NMS.

**Inflexible representation of bounding boxes:** The widely used bounding box representation can be viewed as Dirac delta distribution of the target box coordinates. In fact, the real distribution can be more arbitrary and flexible.

The article proposed new representations for the bounding boxes and their localization quality:

- Merging the localization quality representation with the classification score into a single and unified representation: a classification vector where its value at the ground-truth category index refers to its corresponding localization quality (typically the IoU score between the predicted box and the corresponding ground-truth box).
- Representing the arbitrary distribution for bounding box representation by directly learning the discredited probability distribution over its continuous space.

### Backbone
We use an anchor-free object detection model with a network architecture similar to FCOS:

<img src="resources/backbone.png" align="middle"/>

### Generalized Focal Loss

Generalized Focal Loss (right) is inherited from original Focal Loss (left), however, the current form of Focal Loss only supports {1, 0} discrete labels, but the new labels contain decimals. Therefore, the article proposed to extend the two parts of Focal Loss for enabling the successful training under the case of joint representation:

<img src="resources/gfl.png" align="middle"/>

**Quality Focal Loss (QFL):** Merges classification score and IoU quality to be one `cls-iou` score. The target is dynamically updated online and it is in (0, 1]. For negative samples, the target is 0. So during training, not only the quality of the good predictions get trained with label 1, but the quality of all predicted boxes gets supervision.

**Distributional Focal Loss (DFL):** Directly optimizes a distribution of `bbox` boundaries. The regression target is quantized into `n` (in this repository is 17) bins. The target is expressed as the integral over the distribution. The regression target is actually prepared in the same way as FCOS. In essence, instead of predicting 4 distances (from anchor point to 4 edges of `bbox`), now it predicts `4xn`, distribution classification instead of direct regression.

**GIoU Loss** actually should work quite well.
