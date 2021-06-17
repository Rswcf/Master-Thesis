# Appliacation of machine vision based on classical image processing algorithms and machine learning in quality monitoring of the linear winding process
## Table of Contents
* [Project Description](#Project-Description)
* [Deep Learning Pipeline](#Deep-Learning-Pipeline)
    * [Dataset](#Dataset)
    * [Models](#Models)
    * [Performance](#Performance)
* [Machine Learning Pipeline based on classical image processing algorithms](#Machine-Learning-Pipeline-based-on-classical-image-processing-algorithms)
    * [Dataset](#Dataset)
    * [Models](#Models)

## Project Description

The project taps the potential for machine learning in quality monitoring during linear winding process of electric motor production. The goal is to investigate: **to what extent the winding faults during the production can be correctly identified and classified with CNNs** and **whether the faults can be predicted with machine learning models**. The work can be divided in two steps. In the first step a training dataset with about 38400 images are labeled according to winding fault types. The second step includes the following: 
* ***Deep Learning Pipeline:*** Development and evaluation of a winding fault detection pipeline using different types of CNN through transfer learning
* ***Machine Learning Pipeline based on classical image processing algorithms:*** Detection of caster angles from 2D images using OpenCV and investigate whether the faults can be correctly predicted based on the values using different machine learning algorithms.

[logo]: https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Project Description"

## Deep Learning Pipeline

incert picture to describe the pipeline

### Dataset
#### Image Data

The original image data for the project came from the work of previous students in the institute. The whole image dataset consists of 672000 images, from which I selected and labeled approximately 38100 images to build a training dataset for the CNN Networks and the machine leaning pipelines. The images used are shown below:

#### Label Data

To meet the needs of different use cases, I labeled the images in **four forms**. The first aproach is to transform the problem into a **binary classification problem** based on whether or not there is a winding error in the picture. 

| Label | Winding Fault |
| ------ | ------ |
| 0 | no fault |
| 1 | fault |

The second approach is to transform the problem into **multi-class classification** and each image can only be assigned to one class.

| Label | Winding Fault |
| ------ | ------ |
| 0_iO | no fault |
| 1_DW | double winding |
| 2_Luecke | gap |
| 3_Kreuzung | crossover |
| 4_DW_&_Luecke | double winding + gap |
| 5_DW_&_Kreuzung | double winding + crossover |
| 6_Luecke_&_Kreuzung | gap + crossover |
| 7_DW_&_Luecke _&_Kreuzung | double winding + gap + crossover |

The third and fourth aproaches are to cast the models to **multi-label** and **multi-output classifications**. In these two forms each image can have serveral labels.

| Label | Winding Fault |
| ------ | ------ |
| 0_iO | no fault |
| 1_DW | double winding |
| 2_Luecke | gap |
| 3_Kreuzung | crossover |

### Models
For fast modeling I utilized the first layers of the three pre-trained models respectively through **transfer learning** (e.g. **InceptionV3**, **VGG16** and **ResNet50**) from Keras to reuse the learned features. Totally I built **3 pre-trained models * 4 forms = 12 DL models** to test the potentials of different transformation possibilities. Some important hyperparameters of the models are shown below:
| Label Form | Loss Function | Optimizer | Metrics |
| ------ | ------ | ------ | ------ |
| binary | BinaryCrossentropy  | RMSprop | BinaryAccuracy |
| multi-Class | CategoricalCrossentropy  | RMSprop | CategoricalAccuracy |
| multi-label | BinaryCrossentropy | RMSprop | F1-Score |
| multi-output | BinaryCrossentropy | RMSprop | BinaryAccuracy |

### Performance
| Label Form | Model Performance |
| ------ | ------ |
| binary | f1-score > 0.995 for all labels  |
| multi-Class | f1-score < 0.9 for some classes because of inbalanced dataset  |
| multi-label | f1-score > 0.99 for all labels |
| multi-output | f1-score > 0.99 for all labels |

## Machine Learning Pipeline based on classical image processing algorithms
### Dataset
I wrote a script with OpenCV using **traditional image processing algorithms** to extract the castor angle values during the linear winding process were from the image data. Then the castor angle values were used as features to diverse ML-aprroaches. As labels I choosed only the binary form.
### Models
After feature engineering, I used six machine learning classifiers to test the predictability of faults.

## Files
