# Image Classification with ResNet

This project uses the ResNet deep learning model for image classification.

## Overview

ResNet (Residual Neural Network) was introduced in the 2015 paper ["Deep Residual Learning for Image Recognition"](https://arxiv.org/abs/1512.03385) by Kaiming He et al.

Key aspects of ResNet:

- Addresses vanishing gradients problem in very deep networks using skip connections.
- Skip connections provide a direct path for gradients to propagate to earlier layers.
- This allows training very deep networks with 50, 100, 150+ layers successfully.
- Achieved state-of-the-art results in image classification tasks.

## Usage

Steps to use ResNet for image classification:

- Import ResNet50 and other needed packages.
- Preprocess input image by resizing and converting to array.
- Apply input preprocessing using `preprocess_input()`.
- Load pretrained ResNet50 model with `weights='imagenet'`.
- Do prediction using `model.predict(input_image)`.
- Decode output class index into class label.

## Results

ResNet models produce state-of-the-art results on image classification benchmarks like ImageNet.

On a sample dataset, ResNet50 obtains ~96% accuracy in classifying different objects like dogs, cats etc.

## References

["Deep Residual Learning for Image Recognition"](https://arxiv.org/abs/1512.03385)

