#############################################################################
# Original code copyright 2019 Divam Gupta, modified and used with GPL 3.0  #
# https://github.com/divamgupta/image-segmentation-keras                    #
#############################################################################

import os
import cv2
import six
import glob
import json
import random
import operator
import itertools
import numpy as np
import pandas as pd

from tqdm import tqdm

import keras
from keras.models import *
from keras.layers import *
from types import MethodType

import tensorflow as tf

from sklearn.metrics import jaccard_score
from skimage.transform import resize  # Resize images

IMAGE_ORDERING = 'channels_last'
pretrained_url = "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
# pretrained_url = "C:\\Users\\theza\\Documents\\Uni\\MIT\\2019\\TP\\Project\\Meal-Compliance-Project\\Image-segmentation\\models\\base_model.h5"
class_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(5000)]
MERGE_AXIS = -1
model_from_name = {}


# models/model_utils.py
def get_segmentation_model(input, output):
	img_input = input
	o = output

	o_shape = Model(img_input, o).output_shape
	i_shape = Model(img_input, o).input_shape

	output_height = o_shape[1]
	output_width = o_shape[2]
	input_height = i_shape[1]
	input_width = i_shape[2]
	n_classes = o_shape[3]
	o = (Reshape((output_height * output_width, -1)))(o)

	o = (Activation('softmax'))(o)
	model = Model(img_input, o)
	model.output_width = output_width
	model.output_height = output_height
	model.n_classes = n_classes
	model.input_height = input_height
	model.input_width = input_width
	model.model_name = ""

	model.train = MethodType(train, model)
	model.predict_segmentation = MethodType(predict, model)
	# model.predict_multiple = MethodType( predict_multiple , model )
	# model.evaluate_segmentation = MethodType( evaluate , model )

	return model


def image_segmentation_generator(images_path, segs_path, batch_size, n_classes, input_height, input_width,
                                 output_height, output_width, do_augment=False):
	img_seg_pairs = get_pairs_from_paths(images_path, segs_path)
	random.shuffle(img_seg_pairs)
	zipped = itertools.cycle(img_seg_pairs)

	while True:
		X = []
		Y = []
		for _ in range(batch_size):
			im, seg = next(zipped)

			im = cv2.imread(im, 1)
			seg = cv2.imread(seg, 1)

			X.append(get_image_arr(im, input_width, input_height, odering=IMAGE_ORDERING))
			Y.append(get_segmentation_arr(seg, n_classes, output_width, output_height))

		yield np.array(X), np.array(Y)


def train(model,
          train_images,
          train_annotations,
          input_height=None,
          input_width=None,
          n_classes=None,
          verify_dataset=True,
          checkpoints_path=None,
          epochs=5,
          batch_size=2,
          validate=False,
          val_images=None,
          val_annotations=None,
          val_batch_size=2,
          auto_resume_checkpoint=False,
          load_weights=None,
          steps_per_epoch=512,
          optimizer_name='adadelta',
          m=['accuracy'],
          l='categorical_crossentropy'
          ):
	if isinstance(model, six.string_types):  # check if user gives model name insteead of the model object
		# create the model from the name
		assert (not n_classes is None), "Please provide the n_classes"
		if (not input_height is None) and (not input_width is None):
			model = model_from_name[model](n_classes, input_height=input_height, input_width=input_width)
		else:
			model = model_from_name[model](n_classes)

	n_classes = model.n_classes
	input_height = model.input_height
	input_width = model.input_width
	output_height = model.output_height
	output_width = model.output_width

	if validate:
		assert not (val_images is None)
		assert not (val_annotations is None)

	if not optimizer_name is None:
		model.compile(loss=l,
		              optimizer=optimizer_name,
		              metrics=m)

	if not checkpoints_path is None:
		open(checkpoints_path + "_config.json", "w").write(json.dumps({
			"model_class": model.model_name,
			"n_classes": n_classes,
			"input_height": input_height,
			"input_width": input_width,
			"output_height": output_height,
			"output_width": output_width
		}))

	if (not (load_weights is None)) and len(load_weights) > 0:
		print("Loading weights from ", load_weights)
		model.load_weights(load_weights)

	if verify_dataset:
		print("Verifying train dataset")
		verify_segmentation_dataset(train_images, train_annotations, n_classes)
		if validate:
			print("Verifying val dataset")
			verify_segmentation_dataset(val_images, val_annotations, n_classes)

	train_gen = image_segmentation_generator(train_images, train_annotations, batch_size, n_classes, input_height,
	                                         input_width, output_height, output_width)

	if validate:
		val_gen = image_segmentation_generator(val_images, val_annotations, val_batch_size, n_classes, input_height,
		                                       input_width, output_height, output_width)

	if not validate:
		for ep in range(epochs):
			print("Starting Epoch ", ep)
			model.fit_generator(train_gen, steps_per_epoch, epochs=1, use_multiprocessing=False)
			if not checkpoints_path is None:
				model.save_weights(checkpoints_path + "." + str(ep))
				print("saved ", checkpoints_path + ".model." + str(ep))
			print("Finished Epoch", ep)
	else:
		for ep in range(epochs):
			print("Starting Epoch ", ep)
			model.fit_generator(train_gen, steps_per_epoch, validation_data=val_gen, validation_steps=200, epochs=1,
			                    use_multiprocessing=False)
			if not checkpoints_path is None:
				model.save_weights(checkpoints_path + "." + str(ep))
				print("saved ", checkpoints_path + ".model." + str(ep))
			print("Finished Epoch", ep)


def predict(model=None, inp=None, out_fname=None, checkpoints_path=None):
	assert (not inp is None)
	assert ((type(inp) is np.ndarray) or isinstance(inp,
	                                                six.string_types)), "Inupt should be the CV image or the input file name"

	if isinstance(inp, six.string_types):
		inp = cv2.imread(inp)

	assert len(inp.shape) == 3, "Image should be h,w,3 "
	orininal_h = inp.shape[0]
	orininal_w = inp.shape[1]

	output_width = model.output_width
	output_height = model.output_height
	input_width = model.input_width
	input_height = model.input_height
	n_classes = model.n_classes

	x = get_image_arr(inp, input_width, input_height, odering=IMAGE_ORDERING)
	pr = model.predict(np.array([x]))[0]
	pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)

	seg_img = np.zeros((output_height, output_width, 3))
	colors = class_colors

	for c in range(n_classes):
		seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
		seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
		seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')

	seg_img = cv2.resize(seg_img, (orininal_w, orininal_h))

	if not out_fname is None:
		cv2.imwrite(out_fname, seg_img)
	return pr


# models/vgg16.py


def get_vgg_encoder(input_height=224, input_width=224, pretrained='imagenet'):
	assert input_height % 32 == 0
	assert input_width % 32 == 0

	img_input = Input(shape=(input_height, input_width, 3))

	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING)(
		img_input)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING)(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING)(x)
	f1 = x
	# Block 2
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING)(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING)(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING)(x)
	f2 = x

	# Block 3
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING)(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING)(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING)(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING)(x)
	f3 = x

	# Block 4
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING)(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING)(x)
	f4 = x

	# Block 5
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING)(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING)(x)
	f5 = x

	if pretrained == 'imagenet':
		VGG_Weights_path = keras.utils.get_file(pretrained_url.split("/")[-1], pretrained_url)
		Model(img_input, x).load_weights(VGG_Weights_path)

	return img_input, [f1, f2, f3, f4, f5]


def verify_segmentation_dataset(images_path, segs_path, n_classes):
	img_seg_pairs = get_pairs_from_paths(images_path, segs_path)

	assert len(img_seg_pairs) > 0, "Dataset looks empty or path is wrong "

	for im_fn, seg_fn in tqdm(img_seg_pairs):
		img = cv2.imread(im_fn)
		seg = cv2.imread(seg_fn)

		assert (img.shape[0] == seg.shape[0] and img.shape[1] == seg.shape[
			1]), "The size of image and the annotation does not match or they are corrupt " + im_fn + " " + seg_fn
		assert (np.max(seg[:, :, 0]) < n_classes), "The pixel values of seg image should be from 0 to " + str(
			n_classes - 1) + " . Found pixel value " + str(np.max(seg[:, :, 0]))

	print("Dataset verified! ")


def get_pairs_from_paths(images_path, segs_path):
	images = glob.glob(os.path.join(images_path, "*.jpg")) + glob.glob(os.path.join(images_path, "*.png")) + glob.glob(
		os.path.join(images_path, "*.jpeg"))
	segmentations = glob.glob(os.path.join(segs_path, "*.png"))

	segmentations_d = dict(zip(segmentations, segmentations))

	ret = []

	for im in images:
		seg_bnme = os.path.basename(im).replace(".jpg", ".png").replace(".jpeg", ".png")
		seg = os.path.join(segs_path, seg_bnme)
		assert (seg in segmentations_d), (
				im + " is present in " + images_path + " but " + seg_bnme + " is not found in " + segs_path + " . Make sure annotation image are in .png")
		ret.append((im, seg))

	return ret


def get_image_arr(path, width, height, imgNorm="sub_mean", odering='channels_first'):
	if type(path) is np.ndarray:
		img = path
	else:
		img = cv2.imread(path, 1)

	if imgNorm == "sub_and_divide":
		img = np.float32(cv2.resize(img, (width, height))) / 127.5 - 1
	elif imgNorm == "sub_mean":
		img = cv2.resize(img, (width, height))
		img = img.astype(np.float32)
		img[:, :, 0] -= 103.939
		img[:, :, 1] -= 116.779
		img[:, :, 2] -= 123.68
		img = img[:, :, ::-1]
	elif imgNorm == "divide":
		img = cv2.resize(img, (width, height))
		img = img.astype(np.float32)
		img = img / 255.0

	if odering == 'channels_first':
		img = np.rollaxis(img, 2, 0)
	return img


def get_segmentation_arr(path, nClasses, width, height, no_reshape=False):
	seg_labels = np.zeros((height, width, nClasses))

	if type(path) is np.ndarray:
		img = path
	else:
		img = cv2.imread(path, 1)

	img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
	img = img[:, :, 0]

	for c in range(nClasses):
		seg_labels[:, :, c] = (img == c).astype(int)

	if no_reshape:
		return seg_labels

	seg_labels = np.reshape(seg_labels, (width * height, nClasses))
	return seg_labels


def _unet(n_classes, encoder, l1_skip_conn=True, input_height=416, input_width=608):
	img_input, levels = encoder(input_height=input_height, input_width=input_width)
	[f1, f2, f3, f4, f5] = levels

	o = f4

	o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
	o = (Conv2D(512, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
	o = (BatchNormalization())(o)

	o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
	o = (concatenate([o, f3], axis=MERGE_AXIS))
	o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
	o = (Conv2D(256, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
	o = (BatchNormalization())(o)

	o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
	o = (concatenate([o, f2], axis=MERGE_AXIS))
	o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
	o = (Conv2D(128, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
	o = (BatchNormalization())(o)

	o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)

	if l1_skip_conn:
		o = (concatenate([o, f1], axis=MERGE_AXIS))

	o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
	o = (Conv2D(64, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
	o = (BatchNormalization())(o)

	o = Conv2D(n_classes, (3, 3), padding='same', data_format=IMAGE_ORDERING)(o)

	model = get_segmentation_model(img_input, o)

	return model


def vgg_unet(n_classes, input_height=416, input_width=608, encoder_level=3):
	model = _unet(n_classes, get_vgg_encoder, input_height=input_height, input_width=input_width)
	model.model_name = "vgg_unet"
	return model


model_from_name["unet"] = _unet
model_from_name["vgg_unet"] = vgg_unet


def res(annot, heightNew, width):
	annotNew = resize(annot, (heightNew, width), mode='edge', anti_aliasing=False,
	                  anti_aliasing_sigma=None, preserve_range=True,
	                  order=0).astype(int)
	df = (pd.DataFrame(annotNew))
	_, b = pd.factorize(df.values.T.reshape(-1, ))

	# print(df.apply(lambda x: pd.Categorical(x, b).codes).values.shape)
	annotNewOut = df.apply(lambda x: pd.Categorical(x, b).codes).values
	return annotNewOut


def evaluate(model=None, inp_images=None, annotations=None):
	ious = []
	for im, an in tqdm(zip(inp_images, annotations)):
		img_true = res(cv2.cvtColor(cv2.imread(an), cv2.COLOR_BGR2GRAY), 528, 800)
		img_pred = predict(model, im)
		img_true = np.array(img_true).ravel()
		img_pred = np.array(img_pred).ravel()
		iou = jaccard_score(img_true, img_pred, average='micro')
		ious.append(iou)
	return np.mean(ious)


def getUniqueCountFormImage(im):
	img = np.array(im).flatten()
	y = np.bincount(img)
	ii = np.nonzero(y)[0]
	return dict(sorted(dict(zip(ii, y[ii])).items(), key=lambda kv: kv[1]))


def reduceSSCNoise(y_pred, thresh=0.001):
	df = pd.DataFrame.from_dict(getUniqueCountFormImage(y_pred), orient='index').astype(float)
	div = df.iloc[:, 0].values.sum()
	df = df.apply(lambda r: r / div, axis=1)
	dfout = df.to_dict()[0]
	dfout = {k: v for k, v in dfout.items() if v > thresh}
	return np.array(list(dfout.keys()))


def SSC(y_true, y_pred):
	if (y_true.ndim > 1):
		y_true = np.array(y_true).ravel()
	if (y_pred.ndim > 1):
		y_pred = np.array(y_pred).ravel()

	yTestUn = pd.Series(y_true).unique()
	yPredUn = pd.Series(y_pred).unique()
	maxLen = float(max(len(yTestUn), len(yPredUn)))
	numEqual = float(len(set(list(yTestUn)) - (set(list(yTestUn)) - set(list(yPredUn)))))
	return float(numEqual / maxLen)


def SSCLoss(y_true, y_pred):
	y_true = K.flatten(y_true)
	y_pred = K.flatten(y_pred)
	yTestUn, idx = tf.unique(y_true)
	yPredUn, idx = tf.unique(y_pred)
	maxLen = tf.math.maximum(tf.size(yTestUn), tf.size(yPredUn))
	numEqual = tf.size(tf.sets.set_intersection(tf.dtypes.cast(yTestUn, tf.uint16), tf.dtypes.cast(yPredUn, tf.uint16)))
	return tf.math.subtract(tf.constant(1), tf.math.divide(numEqual, maxLen))


def IoU(y_true, y_pred):
	if (y_true.ndim > 1):
		y_true = np.array(y_true).ravel()
	if (y_pred.ndim > 1):
		y_pred = np.array(y_pred).ravel()
	img_true = np.array(y_true).ravel()
	img_pred = np.array(y_pred).ravel()
	return jaccard_score(img_true, img_pred, average='micro')


def IoULoss(y_true, y_pred):
	return 1 - IoU(y_true, y_pred)


def evaluateOne(model=None, inp_images=None, annotations=None, h=528, w=800):
	ious = []
	uniqueScore = []
	uniqueScoreRN = []
	for im, an in zip(inp_images, annotations):
		img_true = res(cv2.cvtColor(cv2.imread(an), cv2.COLOR_BGR2GRAY), h / 2, w / 2)
		img_pred = predict(model, im)
		img_true = np.array(img_true).ravel()
		img_pred = np.array(img_pred).ravel()
		iou = jaccard_score(img_true, img_pred, average='micro')
		ious.append(iou)
		us = SSC(img_true, img_pred)
		usRN = SSC(img_true, reduceSSCNoise(img_pred))
		uniqueScore.append(us)
		uniqueScoreRN.append(usRN)
	return np.mean(ious), np.mean(uniqueScore), np.mean(uniqueScoreRN)
