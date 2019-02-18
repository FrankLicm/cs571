# ========================================================================
# Copyright 2019 Emory University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================
import os
from typing import List, Tuple
from elit.component import Component
from elit.embedding import FastText
from src.util import tsv_reader
from keras.layers import Embedding, Input, TimeDistributed, SpatialDropout1D, Conv1D, GlobalMaxPooling1D,Activation,Concatenate, Dropout
from keras.layers.merge import Concatenate, dot
from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.optimizers import *
from keras.layers.convolutional import *
from keras import regularizers
from keras.layers.core import Reshape, Permute, Dense, Lambda
import numpy as np
import logging
import time
from keras.models import load_model, save_model
from keras import backend as K
from keras.engine.topology import Layer
from keras.callbacks import ModelCheckpoint

class SentimentAnalyzer(Component):
    def __init__(self, resource_dir: str, embedding_file='fasttext-200-180614.bin'):
        """
        Initializes all resources and the model.
        :param resource_dir: a path to the directory where resource files are located.
        """
        self.vsm = FastText(os.path.join(resource_dir, embedding_file))
        # TODO: to be filled.

        self.filters = 256
        self.embedding_size = 200
        self.sentence_length = 267
        self.nb_classes = 5
        self.model_path = os.path.join(resource_dir, 'hw2-model')
        self.epochs = 4
        self.batch_size = 32
        self.kernel_size = 3
        self.nb_hidden_unit = 200
        self.dropout = 0.5

        self.trn_data = tsv_reader(resource_dir, 'sst.trn.tsv')
        self.dev_data = tsv_reader(resource_dir, 'sst.dev.tsv')
        self.tst_data = tsv_reader(resource_dir, 'sst.tst.tsv')

        self.word_dict = {}
        for _, sentence in self.trn_data+self.dev_data+self.tst_data:
            words = sentence.split(" ")
            for word in words:
                if word not in self.word_dict:
                    self.word_dict[word] = len(self.word_dict)
        self.vocabulary_size = len(self.word_dict)

        self.embedding_matrix = np.zeros((self.vocabulary_size, self.embedding_size))
        for word in self.word_dict:
            if self.word_dict[word] >= self.vocabulary_size:
                break
            embedding_vector = self.vsm.model.get_word_vector(word)
            if (embedding_vector is not None) and len(embedding_vector) > 0:
                self.embedding_matrix[self.word_dict[word]] = embedding_vector
        self.build_model()

    def load(self, model_path: str, **kwargs):
        """
        Load the pre-trained model.
        :param model_path:
        :param kwargs:
        """
        # TODO: to be filled
        self.model.load_weights(model_path)


    def save(self, model_path: str, **kwargs):
        """
        Saves the current model to the path.
        :param model_path:
        :param kwargs:
        """
        # TODO: to be filled
        self.model.save(model_path)

    def attention(self, sentence):
        trans_sentence = K.permute_dimensions(sentence, (0, 2, 1))
        match_score = K.tanh(dot([sentence, trans_sentence], (2, 1)))
        sentence_to_sentence_att = K.softmax(K.sum(match_score, axis=-1))
        b_sum = K.sum(sentence_to_sentence_att, axis=1)
        _b_sum = K.expand_dims(b_sum, 1)
        beta = sentence_to_sentence_att / _b_sum
        que_vector = dot([trans_sentence, beta], (2, 1))
        return que_vector

    def train(self, trn_data: List[Tuple[int, List[str]]], dev_data: List[Tuple[int, List[str]]], *args, **kwargs):
        """
        Trains the model.
        :param trn_data: the training data.
        :param dev_data: the development data.
        :param args:
        :param kwargs:
        :return:
        """
        trn_ys, trn_xs = zip(*[(y, self.vsm.emb_list(x)) for y, x in trn_data])
        dev_ys, dev_xs = zip(*[(y, self.vsm.emb_list(x)) for y, x in dev_data])
        trn_xss = []
        for _, sentence in trn_data:
            words = sentence.split(" ")
            s_vec = np.zeros(self.sentence_length)
            for i in range(min(len(words), self.sentence_length)):
                if words[i] in self.word_dict:
                    s_vec[i] = self.word_dict[words[i]]
            trn_xss.append(s_vec)
        dev_xss = []
        for _, sentence in dev_data:
            words = sentence.split(" ")
            s_vec = np.zeros(self.sentence_length)
            for i in range(min(len(words), self.sentence_length)):
                if words[i] in self.word_dict:
                    s_vec[i] = self.word_dict[words[i]]
            dev_xss.append(s_vec)

        checkpoint = ModelCheckpoint(self.model_path, monitor='val_acc', verbose=1, save_best_only=True,
                                     mode='max', save_weights_only=True)
        self.model.fit(np.array(trn_xss), np.array(trn_ys),
                       batch_size=self.batch_size, epochs=self.epochs,
                       callbacks=[checkpoint], validation_data=(np.array(dev_xss), np.array(dev_ys)), verbose=1)

    def build_model(self):
        inputs = Input(shape=(self.sentence_length,), dtype='float32')
        embedding_layer_setence = Embedding(self.vocabulary_size, self.embedding_size,
                                            weights=[self.embedding_matrix],
                                            input_length=self.sentence_length)

        embedding_sentence = embedding_layer_setence(inputs)
        print(np.shape(embedding_sentence))
        rnn_layer = LSTM(self.nb_hidden_unit, return_sequences=True, activation='tanh', dropout=self.dropout)
        bi_rnn = Bidirectional(rnn_layer, merge_mode='sum')(embedding_sentence)
        print(np.shape(bi_rnn))
        conv = Conv1D(self.filters, self.kernel_size, padding='valid', activation='relu')(bi_rnn)
        print(np.shape(conv))
        att_vector = Lambda(self.attention, output_shape=(self.filters,))(conv)
        print(np.shape(att_vector))
        pool = GlobalMaxPooling1D()(conv)
        print(np.shape(pool))
        # rnn_layer = LSTM(self.nb_hidden_unit, activation='tanh', dropout=self.dropout)
        # bi_rnn = Bidirectional(rnn_layer, merge_mode='concat')(conv)
        merge_vectors = Concatenate()([att_vector, pool])
        classes = Dense(units=self.nb_classes, activation='softmax')(merge_vectors)
        self.model = Model(inputs=inputs, outputs=classes)
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def decode(self, data: List[Tuple[int, List[str]]], **kwargs) -> List[int]:
        """
        :param data:
        :param kwargs:
        :return: the list of predicted labels.
        """
        tst_xss = []
        for _, sentence in data:
            words = sentence.split(" ")
            s_vec = np.zeros(self.sentence_length)
            for i in range(min(len(words), self.sentence_length)):
                if words[i] in self.word_dict:
                    s_vec[i] = self.word_dict[words[i]]
            tst_xss.append(s_vec)
        predictions = self.model.predict(np.array(tst_xss))
        return [np.argmax(i) for i in predictions]

    def evaluate(self, data: List[Tuple[int, List[str]]], **kwargs) -> float:
        """
        :param data:
        :param kwargs:
        :return: the accuracy of this model.
        """
        gold_labels = [y for y, _ in data]
        auto_labels = self.decode(data)
        print(gold_labels)
        print(auto_labels)
        total = correct = 0
        for gold, auto in zip(gold_labels, auto_labels):
            if gold == auto:
                correct += 1
            total += 1
        return 100.0 * correct / total


if __name__ == '__main__':
    resource_dir = "../res"
    sentiment_analyzer = SentimentAnalyzer(resource_dir)
    trn_data = tsv_reader(resource_dir, 'sst.trn.tsv')
    dev_data = tsv_reader(resource_dir, 'sst.dev.tsv')
    tst_data = tsv_reader(resource_dir, 'sst.tst.tsv')
    sentiment_analyzer.train(trn_data, dev_data)
    #sentiment_analyzer.evaluate(tst_data)
    #sentiment_analyzer.save(os.path.join(resource_dir, 'hw2-model'))
    #sentiment_analyzer.evaluate(dev_data)
