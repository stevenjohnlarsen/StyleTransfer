import sys
import os

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.callbacks import Callback
from scipy.misc import imresize, imsave
import numpy as np


from model import EncoderDecoder
from util import count_num_samples

TRAIN_PATH = 'data'
TARGET_SIZE = (256, 256)
BATCH_SIZE = 4
epochs = 2

datagen = ImageDataGenerator()
gen = datagen.flow_from_directory(TRAIN_PATH, target_size=TARGET_SIZE,
                                  batch_size=BATCH_SIZE, class_mode=None)


def create_gen(img_dir, target_size, batch_size):
    datagen = ImageDataGenerator()
    gen = datagen.flow_from_directory(img_dir, target_size=target_size,
                                      batch_size=batch_size, class_mode=None)

    def tuple_gen():
        for img in gen:
            if img.shape[0] != batch_size:
                continue

            # (X, y)
            yield (img, img)

    return tuple_gen()

# This needs to be in scope where model is defined
class OutputPreview(Callback):
    def __init__(self, model, test_img_path, increment, preview_dir_path):
        test_img = image.load_img(test_img_path)
        test_img = imresize(test_img, (256, 256, 3))
        test_target = image.img_to_array(test_img)
        test_target = np.expand_dims(test_target, axis=0)
        self.test_img = test_target
        self.model = model

        self.preview_dir_path = preview_dir_path

        self.increment = increment
        self.iteration = 0

    def on_batch_end(self, batch, logs={}):
        if (self.iteration % self.increment == 0):
            output_img = self.model.predict(self.test_img)[0]
            fname = '%d.jpg' % self.iteration
            out_path = os.path.join(self.preview_dir_path, fname)
            imsave(out_path, output_img)

        self.iteration += 1


gen = create_gen(TRAIN_PATH, TARGET_SIZE, BATCH_SIZE)

num_samples = count_num_samples(TRAIN_PATH)
steps_per_epoch = num_samples // BATCH_SIZE

target_layer = int(sys.argv[1])

encoder_decoder = EncoderDecoder(target_layer=target_layer)

callbacks = [OutputPreview(encoder_decoder, './doge-256.jpg', 5000, './preview-%d' % target_layer)]
encoder_decoder.model.fit_generator(gen, steps_per_epoch=steps_per_epoch,
        epochs=epochs, callbacks=callbacks)
encoder_decoder.export_decoder()
