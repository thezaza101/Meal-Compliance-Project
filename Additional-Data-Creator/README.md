# Additional Data Creator
 
This program takes in image file and segmentation maps and creates augmented versions. This is to provide additional data for training other models

## Requirements

### Additional for Windows only
- install osgeo4w from https://trac.osgeo.org/osgeo4w/
- add installed folder to PATH e.g. C:\OSGeo4W64\bin (must contain geos_c.dll)
- restart command line
- **ensure python path is in PATH**, this can be done while installing python.

### Installation using pip3
- Assuming Python  >= 3.7
- pip3 install six numpy scipy Pillow matplotlib scikit-image opencv-python imageio Shapely
- pip3 install imgaug

