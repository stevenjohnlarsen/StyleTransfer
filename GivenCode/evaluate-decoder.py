from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from scipy.misc import imsave

from model import EncoderDecoder

DECODER_PATH = './models/decoder_5.h5'
INPUT_IMG_PATH = './doge-256.jpg'
OUTPUT_IMG_PATH = './doge-decoded.jpg'

encoder_decoder = EncoderDecoder(decoder_path=DECODER_PATH)

input_img = image.load_img(INPUT_IMG_PATH)
input_img = image.img_to_array(input_img)
input_img = np.expand_dims(input_img, axis=0)

output_img = encoder_decoder.model.predict([input_img])[0]
imsave(OUTPUT_IMG_PATH, output_img)
