# Image Classification with VGG Models

This project uses the VGG convolutional neural network models (VGG16 and VGG19) for image classification on the ImageNet dataset.

## Overview

The VGG models were proposed in the 2014 paper ["Very Deep Convolutional Networks for Large-Scale Image Recognition"](https://arxiv.org/abs/1409.1556) by K. Simonyan and A. Zisserman.

Key aspects:

- VGG16 contains 16 layers (13 convolutional and 3 fully-connected).
- VGG19 contains 19 layers (16 convolutional and 3 fully-connected).
- Both use small 3x3 filters in convolutional layers.
- Achieved state-of-the-art results on ImageNet classification.

## VGG16

VGG16 provides great generalization capability and feature extraction power for transfer learning.

On ImageNet dataset, VGG16 achieves:

- Top-1 accuracy: 71.3%
- Top-5 accuracy: 90.1%

## VGG19

VGG19 has slightly higher accuracy than VGG16, at the cost of increased model size and computational requirements.

On ImageNet dataset, VGG19 achieves:

- Top-1 accuracy: 72.4%
- Top-5 accuracy: 91.1%

## Usage

To use VGG models for image classification:

1. Import model and preprocess input image.
2. Load pretrained weights with `weights='imagenet'`.
3. Make predictions using `model.predict(input_image)`.
4. Decode predictions into class labels.

## Conclusion

The VGG models provide strong baselines for image classification tasks. Their pretrained weights can be used for transfer learning in computer vision.

## References

[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
