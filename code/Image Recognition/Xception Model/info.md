# Image Classification with Xception

This project uses the Xception model for image classification. Xception was introduced in the 2017 paper ["Xception: Deep Learning with Depthwise Separable Convolutions"](https://arxiv.org/abs/1610.02357) by Francois Chollet.

## Overview

Key points about Xception:

- Proposed by Francois Chollet, author of Keras.
- Builds on the Inception architecture using depthwise separable convolutions.
- These separable convolutions help reduce model parameters and computational cost.
- Slightly outperforms Inception V3 on ImageNet classification.

## Usage

Steps to use Xception for image classification:

1. Import Xception and other required packages.
2. Preprocess input image by resizing and converting to array.
3. Apply input preprocessing using `preprocess_input()`.
4. Load pretrained Xception model with `weights='imagenet'`.
5. Make predictions using `model.predict(input_image)`.
6. Decode prediction index into class label.

## Results

Xception achieves 79.0% top-1 and 94.5% top-5 accuracy on the ImageNet dataset.

It provides efficient and accurate feature extraction for transfer learning.

## References

["Xception: Deep Learning with Depthwise Separable Convolutions"](https://arxiv.org/abs/1610.02357)
