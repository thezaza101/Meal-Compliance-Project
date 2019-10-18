import os
import itertools
import scipy.misc
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum
from scipy import ndimage
from collections import Counter


class Color(Enum):
	Ignore = 1
	Hot = 2
	Cold = 3


cols = {
	Color.Ignore: 100,
	Color.Cold: 128,
	Color.Hot: 256
}

db = pd.read_csv(os.path.abspath(os.path.join('Data','classes.csv')))


def removeSmallBlobs(img, avr, threh=0.01):
	"""This function requrns a black image if the average value of the matrix is lower than the specified threashold"""
	if (threh < avr):
		return img
	else:
		return np.zeros((img.shape[0], img.shape[1]), dtype=np.uint)


def identifyHeatColdIntersections(hmap):
	"""This function retuns the areas of the image that doesnt belong in either the Hot or Cold areas but between them"""
	base = np.zeros((hmap.shape[0], hmap.shape[1]), dtype=np.uint)
	test = np.logical_and(hmap > cols[Color.Cold], hmap < cols[Color.Hot])
	base[test] = 255
	base = ndimage.gaussian_filter(base, sigma=1)
	return base


def determineHeatColdIntersectionIssues(intersection):
	"""This function splits the image in half, image with more hot/cold intersection is correct heat side, other side is assumed to have issues"""
	wh = int(round(intersection.shape[1] / 2))
	h1 = intersection[:, :wh]
	h2 = intersection[:, wh:]
	h1a = np.average(h1)
	h2a = np.average(h2)
	if (h1a > h2a):
		h1 = np.zeros(h1.shape, dtype=np.uint)
		if (h2a > 0):
			h2 = removeSmallBlobs(h2, h2a)
	else:
		h2 = np.zeros(h2.shape, dtype=np.uint)
		if (h1a > 0):
			h1 = removeSmallBlobs(h1, h1a)

	return np.concatenate((h1, h2), axis=1)


def getClassHeatValue(database, inCls):
	"""This function allocates heat a value for the input class"""
	hot = database.loc[database['_id'] == inCls, '_hot'].values[0]
	cold = database.loc[database['_id'] == inCls, '_cold'].values[0]
	if (hot == 1 and cold == 1):
		return Color.Ignore
	if (hot == 1 and cold != 1):
		return Color.Hot
	else:
		return Color.Cold


def generateHeatMap(seg, uni):
	"""This function converts the segmentation map into a heat map"""
	base = np.zeros((seg.shape[0], seg.shape[1]), dtype=np.uint)
	for u in uni:
		if (u != 0):
			test = seg == u
			base[test] = cols[getClassHeatValue(db, u)]
	base = ndimage.gaussian_filter(base, sigma=1)

	return base


def analyseAndGetOutput(segmentMap, uniqueElements):
	""" Utelises the other functions to return any isssues"""
	return determineHeatColdIntersectionIssues(
		identifyHeatColdIntersections(
			generateHeatMap(segmentMap, uniqueElements)))
