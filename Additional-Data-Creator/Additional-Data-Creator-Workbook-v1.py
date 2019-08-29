#!/usr/bin/env python
# coding: utf-8

# # Additional Data Creator Workbook

# In[122]:


# Import the relevent packages
import os # File/directory operations
import ntpath # File/directory operations
import imgaug as ia # Image augmentations
import imgaug.augmenters as iaa # Image augmentations
import imageio as imio # Reading images
import numpy as np # Matrix operations
from skimage import data # Image operations
from skimage.color import rgb2gray # RGB image to grey
from skimage.transform import resize # Resize images
import matplotlib.pyplot as plt # for visualisation
import pandas as pd

from scipy import stats


# In[140]:


# Set the input and output paths for later reference
dataPaths = {'InputImage':os.path.abspath(os.path.join('Data','Images')),
             'InputAnnot':os.path.abspath(os.path.join('Data','Annotations')),
             'OutputImage':os.path.abspath(os.path.join('Output','Images')),
             'OutputAnnot':os.path.abspath(os.path.join('Output','Annotations')),
             'XTrain':os.path.abspath(os.path.join('OutputData','XTrain')),
             'yTrain':os.path.abspath(os.path.join('OutputData','yTrain')),
             'XTest':os.path.abspath(os.path.join('OutputData','XTest')),
             'yTest':os.path.abspath(os.path.join('OutputData','yTest'))}


# In[124]:


def getFilesInDir(dir):
    """This function will return a list of files in the given dir"""
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(dir):
        for file in f:
            files.append(os.path.join(r, file))
    return files

def validateImagePairs(imPairTuple, width):
    """This function reads the image and resizes it to the closest %32=0 for the specified width"""
    imgDir , annotDir = imPairTuple
    img = imio.imread(imgDir)
    annot = rgb2gray(imio.imread(annotDir))*255
    
    widthScaleRatio = width / img.shape[1]
    heightNew = (widthScaleRatio * (img.shape[0]))
    heightNew = heightNew - (heightNew%32)
    
    annotNew = resize(annot, (heightNew, width),mode='edge', anti_aliasing=False,
                               anti_aliasing_sigma=None,preserve_range=True,
                               order=0).astype(int)
    df = (pd.DataFrame(annotNew))
    _, b = pd.factorize(df.values.T.reshape(-1, ))  
    r = df.apply(lambda x: pd.Categorical(x, b).codes).add_suffix('_ID')

    # print(df.apply(lambda x: pd.Categorical(x, b).codes).values.shape)
    annotNewOut = df.apply(lambda x: pd.Categorical(x, b).codes).values
    
    
    return (resize(img, (heightNew, width), anti_aliasing=True),
            annotNewOut)

def augment_seg(imSegTuple, filterSeq):
    """This function applies a 'filter' to the input image and its annotation """
    img , seg = imSegTuple    
    aug_det = filterSeq.to_deterministic() 
    image_aug = aug_det.augment_image( img )
    segmap = ia.SegmentationMapOnImage( seg , nb_classes=len(np.unique(seg)), shape=img.shape )
    segmap_aug = aug_det.augment_segmentation_maps( segmap )
    segmap_aug = segmap_aug.get_arr_int()
    return (image_aug , segmap_aug)

def path_leaf(path):
    """This function gets the file name from a path"""
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def saveAugmentOutputs(augmentImagePairs, originalFileNamePairs, suffex, outImagPath, outAnnotPath):
    """This function saves the augmented images"""
    for i in range(0, len(augmentImagePairs)):
        img = augmentImagePairs[i][0]
        fileName = path_leaf(originalFileNamePairs[i][0])
        fileNameSplit = fileName.split('.')
        imFileName = outImagPath+'\\'+fileNameSplit[0]+suffex+'.'+fileNameSplit[1]
        imio.imsave(imFileName, img)
        print(imFileName)
        annot = augmentImagePairs[i][1]
        fileNameAnnot = path_leaf(originalFileNamePairs[i][1])
        fileNameAnnotSplit = fileNameAnnot.split('.')
        annotFileName = outAnnotPath+'\\'+fileNameAnnotSplit[0]+suffex+'.'+fileNameAnnotSplit[1]
        imio.imsave(annotFileName, annot)
        print(annotFileName)

def ShowAugmentCompare(originalTup, newTup):
    """This function shows a comparason of orginal vs augmented images and annotations"""
    fig, axes = plt.subplots(nrows=2, ncols=2)
    ax = axes.ravel()
    ax[0].imshow(originalTup[0])
    ax[0].set_title("Original image")
    ax[1].imshow(newTup[0])
    ax[1].set_title("New image")
    ax[2].imshow(originalTup[1], cmap='gray')
    ax[2].set_title("Original annotation")
    ax[3].imshow(newTup[1], cmap='gray')
    ax[3].set_title("New annotation")
    plt.tight_layout()
    plt.show()


# In[125]:


# This will create a list of tuples for the input images, it will associate an image with 
# its annotation given the naming convention is followed

print('Reading and resizing input files...')

filePairPaths = list(zip(getFilesInDir(dataPaths['InputImage']), getFilesInDir(dataPaths['InputAnnot'])))
inputImagePairs = []
for tup in filePairPaths:
    inputImagePairs.append(validateImagePairs(tup,1600))
    
print('Read '+str(len(inputImagePairs)*2)+' files.')


# In[126]:


# See https://github.com/aleju/imgaug#example-images for details

print('Createing filters...')
fltr1 = iaa.Sequential([
    iaa.Fliplr(1), # horizontally flip 
])

print('\n')
print(fltr1)

fltr2 = iaa.Sequential([
    iaa.Flipud(1), # vertically flip 
])

print('\n')
print(fltr2)
fltr3 = iaa.Sequential([
    iaa.PerspectiveTransform(scale=0.1, keep_size=True), # Perspective Transform
])

print('\n')
print(fltr3)

fltr4 = iaa.Sequential([
    iaa.PiecewiseAffine(scale=0.05), # Piecewise Affine
])

print('\n')
print(fltr4)

print('\n')
print('Done')


# In[127]:


print('Applying filter 1 to input')
inputAugmentImagePairs1 = []
for tup in inputImagePairs:
    inputAugmentImagePairs1.append(augment_seg(tup, fltr1))
    
print('Applying filter 2 to input')
inputAugmentImagePairs2 = []
for tup in inputImagePairs:
    inputAugmentImagePairs2.append(augment_seg(tup, fltr2))
    
print('Applying filter 3 to input')
inputAugmentImagePairs3 = []
for tup in inputImagePairs:
    inputAugmentImagePairs3.append(augment_seg(tup, fltr3))

print('Applying filter 4 to input')
inputAugmentImagePairs4 = []
for tup in inputImagePairs:
    inputAugmentImagePairs4.append(augment_seg(tup, fltr4))
    
# ShowAugmentCompare(inputImagePairs[0],inputAugmentImagePairs1[0])


# In[134]:


print('Saving output of filter 1')
saveAugmentOutputs(inputAugmentImagePairs1, filePairPaths, 'a', dataPaths['OutputImage'],dataPaths['OutputAnnot'])
print('Saving output of filter 2')
saveAugmentOutputs(inputAugmentImagePairs2, filePairPaths, 'b', dataPaths['OutputImage'],dataPaths['OutputAnnot'])
print('Saving output of filter 3')
saveAugmentOutputs(inputAugmentImagePairs3, filePairPaths, 'c', dataPaths['OutputImage'],dataPaths['OutputAnnot'])
print('Saving output of filter 4')
saveAugmentOutputs(inputAugmentImagePairs4, filePairPaths, 'd', dataPaths['OutputImage'],dataPaths['OutputAnnot'])

print('Saving input as output')
saveAugmentOutputs(inputImagePairs, filePairPaths, 'z', dataPaths['OutputImage'],dataPaths['OutputAnnot'])


# In[137]:


from sklearn.model_selection import train_test_split # setting up the test and train data

print('Spilliting train and test sets')
fileOutputPaths = list(zip(getFilesInDir(dataPaths['OutputImage']), getFilesInDir(dataPaths['OutputAnnot'])))
df = pd.DataFrame(fileOutputPaths, columns=['X', 'y'])
X = df['X']
y = df['y']


# In[170]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


# In[196]:


from shutil import copyfile
import errno

def copyFileToDir(df, path):
    for file in df.values:
        print(file)
        dest = path+'\\'
        print(dest)
        dest = dest+path_leaf(file)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        copyfile(file, dest)


# In[197]:


print('Copying train/test output')
copyFileToDir(X_train,dataPaths['XTrain'])
copyFileToDir(X_test,dataPaths['XTest'])
copyFileToDir(y_train,dataPaths['yTrain'])
copyFileToDir(y_test,dataPaths['yTest'])


# In[ ]:





# In[ ]:




