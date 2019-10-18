# USAGE
# python simple_request.py

# import the necessary packages
import requests
import urllib.request
import numpy as np
import time
import json

# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://localhost:5000/"
PREDICT = "predict"
ANALYSE = "analyse"
IMAGE_PATH = "2002a.jpg"

# load the input image and construct the payload for the request
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

# submit the request
start = time.time()
r = requests.post(KERAS_REST_API_URL+PREDICT, files=payload).json()['map']
end = time.time()
print(end - start)
# ensure the request was sucessful
print(np.asarray(r).shape)

payloadAna = {"map": r}


req = urllib.request.Request(KERAS_REST_API_URL+ANALYSE)
req.add_header('Content-Type', 'application/json; charset=utf-8')
jsondata = json.dumps(payloadAna)
jsondataasbytes = jsondata.encode('utf-8')   # needs to be bytes
req.add_header('Content-Length', len(jsondataasbytes))
start = time.time()
r1 = json.loads(urllib.request.urlopen(req, jsondataasbytes).readline().decode("utf-8"))["pred"]
end = time.time()
print(end - start)
print(np.asarray(r1).shape)
