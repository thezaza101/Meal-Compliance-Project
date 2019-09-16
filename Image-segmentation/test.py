from model import *

model = vgg_unet(33,1056,1600)
model.load_weights('C:\\Users\\theza\\Documents\\Uni\\MIT\\2019\\TP\\Project\\Meal-Compliance-Project\\Image-segmentation\\models\\model_latest.h5')

out = model.predict_segmentation(
    inp='1a.jpg',
    out_fname="11cout.png"
)