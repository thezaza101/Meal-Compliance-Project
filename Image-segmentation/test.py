from model import *
model = vgg_unet(33, 1056, 1600)



def get_model_report():
	epochs = ['vgg_unet_1.0.h5',
	          'vgg_unet_1.1.h5',
	          'vgg_unet_1.2.h5',
	          'vgg_unet_1.3.h5',
	          'vgg_unet_1.4.h5']
	basePath = 'C:\\Users\\theza\\Documents\\Uni\\MIT\\2019\\TP\\Project\\Meal-Compliance-Project\\Additional-Data-Creator\\OutputData'

	trainImPath = '\\XTrain'
	trainAnPath = '\\yTrain'
	trainIm = glob.glob(os.path.join(basePath+trainImPath, "*"))
	trainAn = glob.glob(os.path.join(basePath+trainAnPath, "*"))
	trainIm.sort()
	trainAn.sort()

	testImPath = '\\XTest'
	testAnPath = '\\yTest'
	testIm = glob.glob(os.path.join(basePath + testImPath, "*"))
	testAn = glob.glob(os.path.join(basePath + testAnPath, "*"))
	testIm.sort()
	testAn.sort()

	scores = []
	for e in tqdm(epochs):
		if ('.json' not in e):
			print(e)
			model = vgg_unet(33, 1056, 1600)
			model.load_weights(e)
			trainingAc, trainSim = evaluateOne(model, trainIm, trainAn)
			testingAc, testSim = evaluateOne(model, testIm, testAn)
			result = {'epoch': e.split('.')[-2],
			          'train': trainingAc,
			          'test': testingAc,
               'trainSim':trainSim,
               'testSim':testSim}
			print(result)
			scores.append(result)
	return scores

score = get_model_report()


def get_model_report(weights_path, basePath):
	epochs = glob.glob(os.path.join(weights_path, "*"))
	print('dd')
	print(epochs)
	trainImPath = '/XTrain'
	trainAnPath = '/yTrain'
	trainIm = glob.glob(os.path.join(basePath + trainImPath, "*"))
	trainAn = glob.glob(os.path.join(basePath + trainAnPath, "*"))
	trainIm.sort()
	trainAn.sort()
	print(len(trainIm))
	print(len(trainAn))

	testImPath = '/XTest'
	testAnPath = '/yTest'
	testIm = glob.glob(os.path.join(basePath + testImPath, "*"))
	testAn = glob.glob(os.path.join(basePath + testAnPath, "*"))
	testIm.sort()
	testAn.sort()
	print(len(testIm))
	print(len(testAn))
	scores = []
	for e in tqdm(epochs):
		if ('.json' not in e):
			print(e)
			model = vgg_unet(33, 1056, 1600)
			model.load_weights(e)
			#trainingAc = evaluate(model, trainIm, trainAn)
			testingAc = evaluate(model, testIm, testAn)
			result = {'epoch': e.split('.')[-2],
			          #'train': trainingAc,
			          'test': testingAc}
			print(result)
			scores.append(result)
	return scores
