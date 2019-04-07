#!/usr/bin/python
import tensorflow as tf
import os
import csv
import requests

from config import Config
from model import CaptionGenerator
from dataset import prepare_test_data

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string('phase', 'test',
                       'The phase can be train, eval or test')

tf.flags.DEFINE_boolean('load', False,
                        'Turn on to load a pretrained model from either \
                        the latest checkpoint or a specified file')

tf.flags.DEFINE_string('model_file', './models/289999.npy',
                       'If sepcified, load a pretrained model from this file')

tf.flags.DEFINE_boolean('load_cnn', False,
                        'Turn on to load a pretrained CNN model')

tf.flags.DEFINE_string('cnn_model_file', './vgg16_no_fc.npy',
                       'The file containing a pretrained CNN model')

tf.flags.DEFINE_boolean('train_cnn', False,
                        'Turn on to train both CNN and RNN. \
                         Otherwise, only RNN is trained')

tf.flags.DEFINE_integer('beam_size', 3,
                        'The size of beam search for caption generation')

sess = tf.Session()

class show_and_tell_model():

    def __init__(self):
        self.cache = {}
        os.chdir('./show_attend_and_tell')
        self.config = Config()
        self.config.phase = FLAGS.phase
        self.config.train_cnn = FLAGS.train_cnn
        self.config.beam_size = FLAGS.beam_size


        # testing phase
        self.model = CaptionGenerator(self.config)
        self.model.load(sess, FLAGS.model_file)
        tf.get_default_graph().finalize()

    def run_model(self):
        data, vocabulary = prepare_test_data(self.config)
        self.model.test(sess, data, vocabulary)

    def process_list(self, img_list):
        for img in img_list:
            if img.split('/')[-1] in self.cache:
                continue
            self.download_image(img)
        self.run_model()
        self.update_cache()
        self.clear_results()

    def download_image(self, url):
        img_data = requests.get(url).content
        with open('./test/images/' + url.split('/')[-1] + '.jpg', 'wb') as handler:
            handler.write(img_data)

    def get_result(self, url):
        return self.cache.get(url, None)

    def update_cache(self):
        # os.chdir('./show_attend_and_tell')
        with open('./test/results.csv') as csvf:
            rr = csv.reader(csvf)
            results = list(rr)
            for result in results:
                self.cache[result[2].split('/')[-1].split('.jpg')[0]] = result[1] 
        print(self.cache)

    def clear_results(self):
        for file in os.listdir('./test/images/'):
            os.remove('./test/images/' + file)
        os.remove('./test/results.csv')
