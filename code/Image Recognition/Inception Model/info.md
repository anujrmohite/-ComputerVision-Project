
## Inception-V3 Image Classification

This project uses the Inception-V3 model for image classification.

## Overview

Inception-V3 is a convolutional neural network architecture for image classification, originally published in the 2015 paper ["Rethinking the Inception Architecture for Computer Vision"](https://arxiv.org/abs/1512.00567) by Christian Szegedy, et al. from Google.

The key aspects of Inception-V3:

- Very deep network with 42 layers.
- Utilizes Inception modules which apply convolutions on input in parallel.
- Achieved state-of-the-art accuracy on ImageNet classification task.
- More computationally efficient than previous Inception versions.

## Usage

The key steps for using Inception-V3 for image classification:

- Import InceptionV3 and other needed packages.
- Preprocess input image by resizing and converting to array.
- Apply input preprocessing using `preprocess_input()`.
- Load pretrained InceptionV3 model with `weights='imagenet'`.
- Do prediction using `model.predict(input_image)`
- Decode output and print predicted class label.

## Results

Inception-V3 obtains 78.8% top-1 and 94.4% top-5 accuracy on the ImageNet dataset.

On a sample set of images, it is able to correctly classify different objects like dogs, cats, cars etc.

## References

["Rethinking the Inception Architecture for Computer Vision"](https://arxiv.org/abs/1512.00567)

