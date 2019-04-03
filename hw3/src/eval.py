import os
from time import time
import tensorflow as tf
from keras import backend as K
from src.hw3 import NamedEntityRecognizer
from src.util import tsv_reader

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    config = tf.ConfigProto()
    K.tensorflow_backend.set_session(tf.Session(config=config))
    resource_dir = os.environ.get('RESOURCE')
    tst_data = tsv_reader(resource_dir, 'conll03.eng.tst.tsv')
    start = time()
    named_entity_recognizer = NamedEntityRecognizer(resource_dir)
    named_entity_recognizer.load(os.path.join(resource_dir, 'hw3-model'))
    score = named_entity_recognizer.evaluate(tst_data)
    end = time()
    print(score, end - start)
