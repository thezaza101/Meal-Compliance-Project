# Segment Map Merger

Given the layer output from photoshop, and the classes database, this notebook creates the annotated Y files for training and testing

The input files must be organised as follows:
- **Classes Database** - _Data/classes.csv_
- **X Images** - _Data/Images/..._ The images must have a _.jpg_ extension. The image names must be unique, the full name of a image must not be a part of another image file's name.
- **Photoshop Layer Outputs** - _Data/Annotations/..._ These files must have a _.png_ extension. The file name must be in the following format *[Name of X image] _ [Layer number] _ [Class Name] .png* (with no spaces).
