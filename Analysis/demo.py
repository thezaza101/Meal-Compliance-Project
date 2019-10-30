print('Loading files and images..')

import os # File/directory operations
from model import *
tf.logging.set_verbosity(tf.logging.ERROR)
import ntpath # File/directory operations
import matplotlib
import matplotlib.pyplot as plt
import imageio as imio # Reading images
import glob
from collections import Counter
import itertools
import collections
import time

dataImPaths = {'XTrain':os.path.abspath(os.path.join('Data','Image','XTrain')),
             'XTest':os.path.abspath(os.path.join('Data','Image','XTest')),
             'yTrain':os.path.abspath(os.path.join('Data','Image','yTrain')),
             'yTest':os.path.abspath(os.path.join('Data','Image','yTest'))}


imXTrain = glob.glob(os.path.join(dataImPaths['XTrain'], "*"))
imXTest = glob.glob(os.path.join(dataImPaths['XTest'], "*"))

imXTrain.sort()
imXTest.sort()

imYTrain = glob.glob(os.path.join(dataImPaths['yTrain'], "*"))
imYTest = glob.glob(os.path.join(dataImPaths['yTest'], "*"))

imYTrain.sort()
imYTest.sort()

filePairPathsTrain = list(zip(imXTrain, imYTrain))
filePairPathsTest = list(zip(imXTest, imYTest))

print('Found ' + str(len(filePairPathsTest)) + ' images to validate\n')

def getClassFromID(id):
    data = pd.read_csv('Data/classes.csv')
    return data.loc[data['_id'] == id, '_name'].values[0]


def path_leaf(path):
    """This function gets the file name from a path"""
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

print('Loading model...')
start = time.time()
model = vgg_unet(43, 256, 416)
model.load_weights('Data/weights.h5')
end = time.time()
print('Time to load model: %.2f s' % (end - start))
print('\n')

testPairs = {}
for im in filePairPathsTest:
    print(path_leaf(im[0])+'-'+path_leaf(im[1])+'-------------------------------\n')
        
    true = imio.imread(im[1])                        
    true = np.array(true).ravel()
    
    uniqueTrue = reduceSSCNoise(true,0)
    uniqueTrue.sort()
    
    print('Image cointains classes:')    
    print([getClassFromID(x) for x in uniqueTrue])
    print('\n')
    
    start = time.time()
    out = predict(model, im[0])
    end = time.time()
    out = np.array(out).ravel()                        
    t = 0.001
    uniquePred = reduceSSCNoise(out,t)
    uniquePred.sort()
    
    print('Model predicted classes:')  
    print([getClassFromID(x) for x in uniquePred])
    print('\nTime to predict: %.2f s' % (end - start))
    
    print('SSC: '+ str(SSC(true,reduceSSCNoise(out,t))))
    print('\n')
    
print('Done')