# Additional Data Creator

## Change history
**29-Aug-19:**
- initial draft

**30-Aug-19:**
- orchestrated filters
- now with less memory useage!

**09-Sept-19:**
- Changed filters
- bug fixes

# Overview
This program takes in image file and segmentation maps and creates augmented versions. This is to provide additional data for training other models

One of the biggest issues when training deep neural networks is a lack of data. Using the same data over and over again will lead to overfitting. Fortunately for image data we can create augmented data that will help alleviate some of our data problems. Using this data to train our network will result in a much more robust and generalised model.

## Issues to solve

- Not enough data to train network on
- Training images are from the same angle, this will cause degraded performance if every image isn’t exactly from the same angle
- Network requires images that is divisible by 32

## Terminology

- Image data = RGB image of the meal tray
- Annotation data = Greyscale image of the same size as Image data, each pixel value represents a class in the image. i.e. pixel value of 15 = meat, 30 = soup, etc..

## Filter setup

| **Filter** | **What** | **Why** |
| --- | --- | --- |
| MotionBlur | Motion Blur | Tray Is in motion when picture is taken |
| Fliplr | Flip on Y axis | Tray is not always in the same orientation |
| Flipud | Flip on X axis | Tray is not always in the same orientation |
| PerspectiveTransform | Perspective Transform | Tray may not always be exactly underneath camera |
| PiecewiseAffine | Distort | Tray may not always be exactly underneath camera |
| Multiply 0.5 1.5 | Lighting – Darker brighter | Cater for lighting conditions |
| ChannelShuffle | Lighting hue | Change in lighting conditions |

| **FilterPrefix** | **a** | **b** | **c** | **d** | **e** | **f** | **g** | **h** | **i** | **j** | **k** | **l** | **m** | **n** | **o** | **p** | **q** | **r** | **s** | **t** | **u** | **v** | **w** | **x** | **y** | **z** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MotionBlur |   | X |   |   |   |   |   |   | X | X | X |   |   |   |   |   |   | X |   |   |   | X |   |   |   | X |
| Fliplr |   |   | X |   |   |   |   |   | X | X | X | X |   |   |   |   | X |   | X | X | X |   |   | X |   |   |
| Flipud |   |   |   | X |   |   |   |   |   | X | X | X | X |   |   | X |   |   | X | X | X |   | X |   |   |   |
| PerspectiveTransform |   |   |   |   | X |   |   | X |   |   | X | X | X | X |   | X |   | X |   |   | X |   | X |   |   | X |
| PiecewiseAffine |   |   |   |   |   | X |   | X |   |   |   | X | X | X | X |   |   |   |   | X |   |   |   |   | X |   |
| Multiply 0.5 1.5 |   |   |   |   |   |   | X |   |   |   |   |   | X | X |   |   |   | X | X |   | X |   |   |   | X | X |
| ChannelShuffle |   |   |   |   |   |   |   | X | X |   |   |   |   | X | X | X | X | X | X | X |   | X |   | X |   | X |


## High level steps

1. Read in image and annotation data as a list of tuples
2. Foreach tuple:
   1. resize the width to a specified value (1600 in current script)
   2. calculate the scaling factor compared to the original image
   3. determine height of scaled image (height of original image \* scaling factor)
   4. calculate the closest value to new height that is divisible by 32
   5. resize the image to new width and height value
   6. apply same scaling to the annotation image
      1. **Issue** , when resizing the annotation image we can&#39;t use anti aliasing or interpolation as it interduces new greyscale values. This will lead to images that are more &#39;jagged&#39; – especially if upscaling
   7. Augment each tuple using the same filter (currently have 4 different filters)
   8. Save the output of each filter

## Results

**Input:** 12 images (6 images and 6 annotations)
**Output:** 312 Images (156 images and 156 annotations)

## Requirements

### Additional for Windows only
- install osgeo4w from https://trac.osgeo.org/osgeo4w/
- add installed folder to PATH e.g. C:\OSGeo4W64\bin (must contain geos_c.dll)
- restart command line
- **ensure python path is in PATH**, this can be done while installing python.

### Installation using pip3
- Assuming Python  >= 3.7
- pip3 install sklearn
- pip3 install six numpy scipy Pillow matplotlib scikit-image opencv-python imageio Shapely
- pip3 install imgaug

