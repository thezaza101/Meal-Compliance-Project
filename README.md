# Meal Compliance Project
 
This repository contains work done for the Meal Compliance Project, at the University of Canberra, for the Canberra Hospital.

The project is broken down into 6 sub projects
- **Analysis** - Random bits of code used to analyse model performance and prototype
- **Additional Data Creator** - applies image augmentation to create additional training data
- **Image Preprocessor** - _This system is not complete_, Given image of tray, crop and resize the image to the tray
- **Image Segmentation** - Working files for testing out the modified U-Net model
- **SegmentMapMerger** - Given the layer output from photoshop, and the classes database, create the annotated Y files for training and testing
- **Integration Test** - Flask API that serves the model. This contains the final files

See each project folder for more details

Note that this project uses openly licenced code from [divamgupta](https://github.com/divamgupta/image-segmentation-keras)