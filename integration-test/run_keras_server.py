# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from model import *
from PIL import Image
import numpy as np
import flask
import io

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

graph = None
def load_model():
	# load the pre-trained Keras model (here we are using a model
	# pre-trained on ImageNet and provided by Keras, but you can
	# substitute in your own networks just as easily)
	global model
	model = vgg_unet(43, 256, 416)
	model.load_weights(os.path.abspath(os.path.join('weights.h5')))
	global graph
	graph = tf.get_default_graph()

def prepare_image(image, target):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")

	# resize the input image and preprocess it
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = imagenet_utils.preprocess_input(image)

	# return the processed image
	return image

@app.route("/predict", methods=["POST"])
def predict():
	global graph
	with graph.as_default():
		# initialize the data dictionary that will be returned from the
		# view
		data = {"success": False}

		# ensure an image was properly uploaded to our endpoint
		if flask.request.method == "POST":
			if flask.request.files.get("image"):
				# read the image in PIL format
				image = flask.request.files["image"].read()
				image = np.array(Image.open(io.BytesIO(image)))

				# preprocess the image and prepare it for classification
				# image = prepare_image(image, target=(256, 416, 3))

				# classify the input image and then initialize the list
				# of predictions to return to the client

				preds = model.predict_segmentation(inp=image)
				output = {}
				output['pred'] = preds.tolist()

	# return the data dictionary as a JSON response
	return output

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	load_model()
	app.run()