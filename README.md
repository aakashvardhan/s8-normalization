# Image Classification on CIFAR-10 dataset using CNN with various Normalization techniques

## Introduction

This project is an implementation of a Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset. The CNN is trained using various normalization techniques such as Batch Normalization, Layer Normalization, and Group Normalization. Our goal is to compare the performance of the CNN with different normalization techniques and to understand the impact of normalization on the training and performance, including the number of misclassified images.

## CIFAR-10 Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images. The classes are: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

Here are some sample images from the CIFAR-10 dataset:

![CIFAR-10 Sample Images](https://github.com/aakashvardhan/s8-normalization/blob/main/assets/cifar10-sample-img.png)

RGB Images from CIFAR-10 dataset:

![CIFAR-10 RGB Images](https://github.com/aakashvardhan/s8-normalization/blob/main/assets/download.png)

## Normalization Techniques

Normalization is a technique used to scale the input features to a similar range so that the model can learn effectively.

1. **Batch Normalization**: Batch normalization normalizes the input of each layer by adjusting and scaling the activations. It helps in reducing the internal covariate shift and speeds up the training process.

2. **Layer Normalization**: Layer normalization normalizes the input of each layer by computing the mean and variance of the entire layer. It helps in reducing the internal covariate shift and speeds up the training process.

3. **Group Normalization**: Group normalization normalizes the input of each layer by dividing the channels into groups and computing the mean and variance for each group. It helps in reducing the internal covariate shift and speeds up the training process.

### Batch Normalization

We use **nn.BatchNorm2d** to implement Batch Normalization in the model.

```python
nn.BatchNorm2d(num_features)
```

#### Architecture

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 32, 32]             864
              ReLU-2           [-1, 32, 32, 32]               0
       BatchNorm2d-3           [-1, 32, 32, 32]              64
           Dropout-4           [-1, 32, 32, 32]               0
         ConvBlock-5           [-1, 32, 32, 32]               0
            Conv2d-6           [-1, 32, 32, 32]           9,216
              ReLU-7           [-1, 32, 32, 32]               0
       BatchNorm2d-8           [-1, 32, 32, 32]              64
           Dropout-9           [-1, 32, 32, 32]               0
        ConvBlock-10           [-1, 32, 32, 32]               0
           Conv2d-11           [-1, 16, 32, 32]             512
        MaxPool2d-12           [-1, 16, 16, 16]               0
  TransitionBlock-13           [-1, 16, 16, 16]               0
           Conv2d-14           [-1, 16, 16, 16]           2,304
             ReLU-15           [-1, 16, 16, 16]               0
      BatchNorm2d-16           [-1, 16, 16, 16]              32
          Dropout-17           [-1, 16, 16, 16]               0
        ConvBlock-18           [-1, 16, 16, 16]               0
           Conv2d-19           [-1, 32, 16, 16]           4,608
             ReLU-20           [-1, 32, 16, 16]               0
      BatchNorm2d-21           [-1, 32, 16, 16]              64
          Dropout-22           [-1, 32, 16, 16]               0
        ConvBlock-23           [-1, 32, 16, 16]               0
           Conv2d-24           [-1, 32, 16, 16]           9,216
             ReLU-25           [-1, 32, 16, 16]               0
      BatchNorm2d-26           [-1, 32, 16, 16]              64
          Dropout-27           [-1, 32, 16, 16]               0
        ConvBlock-28           [-1, 32, 16, 16]               0
           Conv2d-29           [-1, 16, 16, 16]             512
        MaxPool2d-30             [-1, 16, 8, 8]               0
  TransitionBlock-31             [-1, 16, 8, 8]               0
           Conv2d-32             [-1, 16, 8, 8]           2,304
             ReLU-33             [-1, 16, 8, 8]               0
      BatchNorm2d-34             [-1, 16, 8, 8]              32
          Dropout-35             [-1, 16, 8, 8]               0
        ConvBlock-36             [-1, 16, 8, 8]               0
           Conv2d-37             [-1, 32, 8, 8]           4,608
             ReLU-38             [-1, 32, 8, 8]               0
      BatchNorm2d-39             [-1, 32, 8, 8]              64
          Dropout-40             [-1, 32, 8, 8]               0
        ConvBlock-41             [-1, 32, 8, 8]               0
           Conv2d-42             [-1, 32, 8, 8]           9,216
             ReLU-43             [-1, 32, 8, 8]               0
      BatchNorm2d-44             [-1, 32, 8, 8]              64
          Dropout-45             [-1, 32, 8, 8]               0
        ConvBlock-46             [-1, 32, 8, 8]               0
        AvgPool2d-47             [-1, 32, 1, 1]               0
           Conv2d-48             [-1, 10, 1, 1]             330
================================================================
Total params: 44,138
Trainable params: 44,138
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 3.71
Params size (MB): 0.17
Estimated Total Size (MB): 3.89
----------------------------------------------------------------
```

#### Training and Testing Results

```
EPOCH: 0
Loss=1.2672936916351318 Batch_id=390 Accuracy=38.41: 100%|██████████| 391/391 [01:02<00:00,  6.30it/s]

Test set: Average loss: 1.4610, Accuracy: 4678/10000 (46.78%)

EPOCH: 1
Loss=1.34928297996521 Batch_id=390 Accuracy=50.16: 100%|██████████| 391/391 [01:03<00:00,  6.15it/s]

Test set: Average loss: 1.2026, Accuracy: 5588/10000 (55.88%)

EPOCH: 2
Loss=1.3156006336212158 Batch_id=390 Accuracy=53.76: 100%|██████████| 391/391 [01:03<00:00,  6.14it/s]

Test set: Average loss: 1.0884, Accuracy: 6093/10000 (60.93%)

EPOCH: 3
Loss=1.227776288986206 Batch_id=390 Accuracy=56.14: 100%|██████████| 391/391 [01:03<00:00,  6.15it/s]

Test set: Average loss: 1.0693, Accuracy: 6115/10000 (61.15%)

EPOCH: 4
Loss=1.1336103677749634 Batch_id=390 Accuracy=58.37: 100%|██████████| 391/391 [01:03<00:00,  6.11it/s]

Test set: Average loss: 1.0295, Accuracy: 6222/10000 (62.22%)

...
...
...
...

EPOCH: 15
Loss=0.9887039065361023 Batch_id=390 Accuracy=65.31: 100%|██████████| 391/391 [01:04<00:00,  6.08it/s]

Test set: Average loss: 0.8291, Accuracy: 7062/10000 (70.62%)

EPOCH: 16
Loss=0.8864884376525879 Batch_id=390 Accuracy=65.69: 100%|██████████| 391/391 [01:04<00:00,  6.02it/s]

Test set: Average loss: 0.8291, Accuracy: 7047/10000 (70.47%)

EPOCH: 17
Loss=0.8401225805282593 Batch_id=390 Accuracy=65.56: 100%|██████████| 391/391 [01:04<00:00,  6.06it/s]

Test set: Average loss: 0.8243, Accuracy: 7073/10000 (70.73%)

EPOCH: 18
Loss=0.9530172348022461 Batch_id=390 Accuracy=65.84: 100%|██████████| 391/391 [01:04<00:00,  6.06it/s]

Test set: Average loss: 0.8216, Accuracy: 7063/10000 (70.63%)

EPOCH: 19
Loss=1.005210518836975 Batch_id=390 Accuracy=65.47: 100%|██████████| 391/391 [01:04<00:00,  6.07it/s]

Test set: Average loss: 0.8287, Accuracy: 7057/10000 (70.57%)

```

![Plot](https://github.com/aakashvardhan/s8-normalization/blob/main/assets/bn-model-performance.png)


#### Misclassified Images from **BatchNorm** Model

![Misclassified Images](https://github.com/aakashvardhan/s8-normalization/blob/main/assets/bn-model-misclass-imgs.png)

### Layer Normalization

The architecture and training/testing results for Layer Normalization are similar to Batch Normalization, except for the normalization technique used.

We use **nn.GroupNorm** to implement Layer Normalization in the model. The group size is set to 1, which is equivalent to Layer Normalization.

```python
class LayerNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.layer_norm = nn.GroupNorm(num_groups=GROUP_SIZE_LN, num_channels=num_features)
        
    def forward(self, x):
        return self.layer_norm(x)
```

#### Training and Testing Results

```
EPOCH: 1
Loss=1.6652904748916626 Batch_id=390 Accuracy=30.59: 100%|██████████| 391/391 [00:55<00:00,  7.01it/s]

Test set: Average loss: 1.6110, Accuracy: 4039/10000 (40.39%)

EPOCH: 2
Loss=1.5776218175888062 Batch_id=390 Accuracy=41.56: 100%|██████████| 391/391 [00:57<00:00,  6.77it/s]

Test set: Average loss: 1.3718, Accuracy: 5001/10000 (50.01%)

EPOCH: 3
Loss=1.348969578742981 Batch_id=390 Accuracy=47.67: 100%|██████████| 391/391 [00:56<00:00,  6.90it/s]

Test set: Average loss: 1.3007, Accuracy: 5296/10000 (52.96%)

EPOCH: 4
Loss=1.497654914855957 Batch_id=390 Accuracy=51.41: 100%|██████████| 391/391 [00:57<00:00,  6.80it/s]

Test set: Average loss: 1.2129, Accuracy: 5669/10000 (56.69%)

EPOCH: 5
Loss=1.1423717737197876 Batch_id=390 Accuracy=54.86: 100%|██████████| 391/391 [00:58<00:00,  6.68it/s]

Test set: Average loss: 1.1367, Accuracy: 5941/10000 (59.41%)

...
...
...
...

EPOCH: 15
Loss=1.1374356746673584 Batch_id=390 Accuracy=64.73: 100%|██████████| 391/391 [00:57<00:00,  6.77it/s]

Test set: Average loss: 0.8426, Accuracy: 7000/10000 (70.00%)

EPOCH: 16
Loss=0.9627411961555481 Batch_id=390 Accuracy=65.31: 100%|██████████| 391/391 [00:58<00:00,  6.66it/s]

Test set: Average loss: 0.8748, Accuracy: 6924/10000 (69.24%)

EPOCH: 17
Loss=0.8431264758110046 Batch_id=390 Accuracy=65.94: 100%|██████████| 391/391 [00:57<00:00,  6.79it/s]

Test set: Average loss: 0.8507, Accuracy: 6972/10000 (69.72%)

EPOCH: 18
Loss=0.6977094411849976 Batch_id=390 Accuracy=66.20: 100%|██████████| 391/391 [00:57<00:00,  6.86it/s]

Test set: Average loss: 0.8304, Accuracy: 7081/10000 (70.81%)

EPOCH: 19
Loss=0.9021633863449097 Batch_id=390 Accuracy=66.45: 100%|██████████| 391/391 [00:58<00:00,  6.70it/s]

Test set: Average loss: 0.8267, Accuracy: 7086/10000 (70.86%)

EPOCH: 20
Loss=1.1069458723068237 Batch_id=390 Accuracy=67.03: 100%|██████████| 391/391 [00:56<00:00,  6.90it/s]

Test set: Average loss: 0.8111, Accuracy: 7167/10000 (71.67%)

```

![Plot](https://github.com/aakashvardhan/s8-normalization/blob/main/assets/ln-model-performance.png)

#### Misclassified Images from **LayerNorm** Model

![Misclassified Images](https://github.com/aakashvardhan/s8-normalization/blob/main/assets/ln-model-misclass-imgs.png)

### Group Normalization

The architecture and training/testing results for Group Normalization are similar to Batch Normalization.

We use **nn.GroupNorm** to implement Group Normalization in the model. The group size is set to 2, which is equivalent to Group Normalization.

```python
GROUP_SIZE_GN = 2

nn.GroupNorm(GROUP_SIZE_GN, num_features)
```

#### Training and Testing Results

```
EPOCH: 1
Loss=1.5534536838531494 Batch_id=390 Accuracy=29.82: 100%|██████████| 391/391 [01:01<00:00,  6.39it/s]

Test set: Average loss: 1.5801, Accuracy: 4097/10000 (40.97%)

EPOCH: 2
Loss=1.5281018018722534 Batch_id=390 Accuracy=42.03: 100%|██████████| 391/391 [01:03<00:00,  6.17it/s]

Test set: Average loss: 1.4073, Accuracy: 4891/10000 (48.91%)

EPOCH: 3
Loss=1.3343065977096558 Batch_id=390 Accuracy=48.38: 100%|██████████| 391/391 [01:02<00:00,  6.25it/s]

Test set: Average loss: 1.3047, Accuracy: 5307/10000 (53.07%)

EPOCH: 4
Loss=1.4690215587615967 Batch_id=390 Accuracy=52.24: 100%|██████████| 391/391 [01:03<00:00,  6.15it/s]

Test set: Average loss: 1.2407, Accuracy: 5534/10000 (55.34%)

EPOCH: 5
Loss=1.08296799659729 Batch_id=390 Accuracy=55.31: 100%|██████████| 391/391 [01:03<00:00,  6.20it/s]

Test set: Average loss: 1.1153, Accuracy: 5970/10000 (59.70%)

...
...
...
...

EPOCH: 15
Loss=1.0328396558761597 Batch_id=390 Accuracy=65.29: 100%|██████████| 391/391 [01:03<00:00,  6.15it/s]

Test set: Average loss: 0.8540, Accuracy: 6917/10000 (69.17%)

EPOCH: 16
Loss=0.9372323155403137 Batch_id=390 Accuracy=66.07: 100%|██████████| 391/391 [01:03<00:00,  6.17it/s]

Test set: Average loss: 0.8233, Accuracy: 7107/10000 (71.07%)

EPOCH: 17
Loss=0.8834392428398132 Batch_id=390 Accuracy=66.47: 100%|██████████| 391/391 [01:03<00:00,  6.13it/s]

Test set: Average loss: 0.8335, Accuracy: 7053/10000 (70.53%)

EPOCH: 18
Loss=0.7396939992904663 Batch_id=390 Accuracy=66.87: 100%|██████████| 391/391 [01:03<00:00,  6.13it/s]

Test set: Average loss: 0.8189, Accuracy: 7106/10000 (71.06%)

EPOCH: 19
Loss=0.9720532298088074 Batch_id=390 Accuracy=67.13: 100%|██████████| 391/391 [01:03<00:00,  6.13it/s]

Test set: Average loss: 0.8030, Accuracy: 7175/10000 (71.75%)

EPOCH: 20
Loss=1.0567971467971802 Batch_id=390 Accuracy=67.63: 100%|██████████| 391/391 [01:04<00:00,  6.10it/s]

Test set: Average loss: 0.8015, Accuracy: 7261/10000 (72.61%)
```

![Plot](https://github.com/aakashvardhan/s8-normalization/blob/main/assets/gn-model-performance.png)

#### Misclassified Images from **GroupNorm** Model

![Misclassified Images](https://github.com/aakashvardhan/s8-normalization/blob/main/assets/gn-model-misclass-imgs.png)

## Conclusion

Comparing the performance of the CNN model with different normalization techniques, we observe the following:

1. **Batch Normalization**: The model achieved an accuracy of 70.57% after 20 epochs. 

2. **Layer Normalization**: The model achieved an accuracy of 70.86% after 20 epochs.

3. **Group Normalization**: The model achieved an accuracy of 72.61% after 20 epochs.


