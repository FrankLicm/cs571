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
import numpy as np
from elit.component import Component
from elit.embedding import FastText
from src.util import tsv_reader
from keras.models import Model
from keras.layers import TimeDistributed, Conv1D, Dense, Embedding, Input, Dropout, LSTM, Bidirectional, MaxPooling1D, \
    Flatten, concatenate
from keras.initializers import RandomUniform
from keras.utils.generic_utils import Progbar
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
import tensorflow as tf
from elit.eval import ChunkF1

class NamedEntityRecognizer(Component):
    def __init__(self, resource_dir: str, embedding_file='fasttext-50-180614.bin'):
        """
        Initializes all resources and the model.
        :param resource_dir: a path to the directory where resource files are located.
        """
        self.vsm = FastText(os.path.join(resource_dir, embedding_file))
        self.embedding_size = 50
        self.epochs = 200
        self.dropout = 0.68
        self.dropout_recurrent = 0.25
        self.lstm_state_size = 275
        self.conv_size = 3
        self.learning_rate = 0.0105
        self.batch_size = 32
        self.model_path = os.path.join(resource_dir, 'hw3-model')
        self.trn_data = tsv_reader(resource_dir, 'conll03.eng.trn.tsv')
        self.dev_data = tsv_reader(resource_dir, 'conll03.eng.dev.tsv')
        self.tst_data = tsv_reader(resource_dir, 'conll03.eng.tst.tsv')
        self.char_length = 0
        self.sentence_length = 0
        self.label_dict = {}
        words = {}
        for labels, sentence in self.trn_data + self.dev_data + self.tst_data:
            for label in labels:
                if label not in self.label_dict:
                    self.label_dict[label] = len(self.label_dict)
            for word in sentence:
                lengc = len(word)
                if lengc > self.char_length:
                    self.char_length = lengc
                words[word.lower()] = True
        self.label_id_dict = {v: k for k, v in self.label_dict.items()}
        self.word_dict = {}
        wordEmbeddings = []

        for word in self.vsm.model.get_words():
            if len(self.word_dict) == 0:  # Add padding+unknown
                self.word_dict["PAD"] = len(self.word_dict)
                vector = np.zeros(self.vsm.dim)  # Zero vector vor 'PAD' word
                wordEmbeddings.append(vector)

                self.word_dict["UNK"] = len(self.word_dict)
                vector = np.random.uniform(-0.25, 0.25, self.vsm.dim)
                wordEmbeddings.append(vector)

            if word.lower() in words:
                vector = self.vsm.model.get_word_vector(word)
                wordEmbeddings.append(vector)
                self.word_dict[word] = len(self.word_dict)

        self.word_embedding_matrix = np.array(wordEmbeddings)
        self.case_embedding_matrix = np.identity(8, dtype='float32')
        self.char_dict = {"PAD": 0, "UNK": 1}
        for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|<>":
            self.char_dict[c] = len(self.char_dict)
        self.build_model()

    def load(self, model_path: str, **kwargs):
        """
        Load the pre-trained model.  .
        :param model_path:
        :param kwargs:
        """
        self.model.load_weights(model_path)

    def save(self, model_path: str, **kwargs):
        """
        Saves the current model to the path.
        :param model_path:
        :param kwargs:
        """
        self.model.save(model_path)

    def train(self, trn_data: List[Tuple[List[str], List[str]]], dev_data: List[Tuple[List[str], List[str]]], *args, **kwargs):
        """
        Trains the model.
        :param trn_data: the training data.
        :param dev_data: the development data.
        :param args:
        :param kwargs:
        :return:
        """
        trn_dataset = self.vectorize(trn_data)
        dev_dataset = self.vectorize(dev_data)
        trn_examples, trn_len = self.generate_batches(trn_dataset)
        dev_examples, dev_len = self.generate_batches(dev_dataset)
        n_updates = 0
        predLabels, correctLabels = self.eval_dataset(dev_examples)
        pre_dev, rec_dev, f1_dev = self.f1(predLabels, correctLabels)
        print('initial Dev accuracy: %.4f %%' % f1_dev)
        best_score =f1_dev
        for epoch in range(self.epochs):
            print("Epoch %d/%d" % (epoch, self.epochs))
            a = Progbar(len(trn_len))
            for i, batch in enumerate(self.generate_minibatches(trn_examples, trn_len)):
                labels, tokens, casing, char = batch
                self.model.train_on_batch([tokens, casing, char], labels)
                a.update(i)
            a.update(i + 1)
            predLabels, correctLabels = self.eval_dataset(dev_examples)
            pre_dev, rec_dev, f1_dev = self.f1(predLabels, correctLabels)
            print('Dev accuracy: %.4f %%' % f1_dev)
            if f1_dev > best_score:
                best_score = f1_dev
                print('Best dev accuracy: epoch = %d, n_udpates = %d, acc = %.4f %%'
                      % (epoch, n_updates, f1_dev))
                self.save(self.model_path)

    def generate_batches(self, data):
        lengths = []
        for i in data:
            lengths.append(len(i[0]))
        lengths = set(lengths)
        batches = []
        batch_len = []
        n = 0
        for i in lengths:
            for batch in data:
                if len(batch[0]) == i:
                    batches.append(batch)
                    n += 1
            batch_len.append(n)
        return batches, batch_len

    def generate_minibatches(self, dataset, batch_len):
        start = 0
        for i in batch_len:
            tokens = []
            casing = []
            char = []
            labels = []
            data = dataset[start:i]
            start = i
            for dt in data:
                t, c, ch, l = dt
                l = np.expand_dims(l, -1)
                tokens.append(t)
                casing.append(c)
                char.append(ch)
                labels.append(l)
            yield np.asarray(labels), np.asarray(tokens), np.asarray(casing), np.asarray(char)

    def vectorize(self, data):
        dataset = []
        for labels, sentence in data:
            word_ids = []
            case_ids = []
            char_ids = []
            label_ids = []
            for word in sentence:
                if word in self.word_dict:
                    word_id = self.word_dict[word]
                elif word.lower() in self.word_dict:
                    word_id = self.word_dict[word.lower()]
                else:
                    word_id = self.word_dict['UNK']
                char_id = []
                chars = [c for c in word]
                for c in chars:
                    if c in self.char_dict:
                        char_id.append(self.char_dict[c])
                word_ids.append(word_id)
                casing = 4
                num_digits = 0
                for char in word:
                    if char.isdigit():
                        num_digits += 1
                digit_fraction = num_digits / float(len(word))
                if word.isdigit():
                    casing = 0
                elif digit_fraction > 0.5:
                    casing = 5
                elif word.islower():
                    casing = 1
                elif word.isupper():
                    casing = 2
                elif word[0].isupper():
                    casing = 3
                elif num_digits > 0:
                    casing = 6
                case_ids.append(casing)
                char_ids.append(char_id)
            for label in labels:
                label_ids.append(self.label_dict[label])
            char_ids = pad_sequences(char_ids, self.char_length, padding='post')
            dataset.append([word_ids, case_ids, char_ids, label_ids])
        return dataset

    def build_model(self):
        words_input = Input(shape=(None,), dtype='int32', name='words_input')
        words = Embedding(input_dim=len(self.word_dict), output_dim=self.vsm.dim,
                          weights=[self.word_embedding_matrix],
                          trainable=False)(words_input)
        casing_input = Input(shape=(None,), dtype='int32', name='casing_input')
        casing = Embedding(output_dim=self.case_embedding_matrix.shape[1],
                           input_dim=self.case_embedding_matrix.shape[0],
                           weights=[self.case_embedding_matrix],
                           trainable=False)(casing_input)
        character_input = Input(shape=(None, self.char_length,), name="Character_input")
        embed_char_out = TimeDistributed(
            Embedding(len(self.char_dict), 30, embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)),
            name="Character_embedding")(
            character_input)
        #char_lstm_out = TimeDistributed(Bidirectional(LSTM(self.char_length,
        #                   return_sequences=True,
        #                   dropout=self.dropout,
        #                   recurrent_dropout=self.dropout_recurrent
        #                   ), name="BiLSTM"))(embed_char_out)
        conv1d_out = TimeDistributed(
            Conv1D(kernel_size=self.conv_size, filters=53, padding='same', activation='tanh', strides=1),
            name="Convolution")(embed_char_out)
        pool_out = TimeDistributed(MaxPooling1D(self.char_length), name="max_pooling")(conv1d_out)
        char = TimeDistributed(Flatten(), name="Flatten")(pool_out)
        char = Dropout(self.dropout)(char)
        output = concatenate([words, casing, char])
        output = Bidirectional(LSTM(self.lstm_state_size,
                                    return_sequences=True,
                                    dropout=self.dropout,
                                    recurrent_dropout=self.dropout_recurrent
                                    ), name="BiLSTM")(output)
        output = TimeDistributed(Dense(len(self.label_dict), activation='softmax'), name="Softmax_layer")(output)
        self.model = Model(inputs=[words_input, casing_input, character_input], outputs=[output])
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer="nadam")

    def eval_dataset(self, dataset):
        correct_labels = []
        pred_labels = []
        for i, data in enumerate(dataset):
            tokens, casing, char, labels = data
            tokens = np.asarray([tokens])
            casing = np.asarray([casing])
            char = np.asarray([char])
            pred = self.model.predict([tokens, casing, char], verbose=False)[0]
            pred = pred.argmax(axis=-1)
            correct_labels.append(labels)
            pred_labels.append(pred)
        return pred_labels, correct_labels

    def decode(self, data: List[Tuple[List[str], List[str]]], **kwargs) -> List[list]:
        """
        :param data:
        :param kwargs:
        :return: the list of predicted labels.
        """
        dataset = self.vectorize(data)
        pred_labels = []
        for i, data in enumerate(dataset):
            tokens, casing, char, labels = data
            tokens = np.asarray([tokens])
            casing = np.asarray([casing])
            char = np.asarray([char])
            pred = self.model.predict([tokens, casing, char], verbose=False)[0]
            pred = pred.argmax(axis=-1)
            pred_labels.append([self.label_id_dict[element] for element in list(pred)])
        return pred_labels

    def evaluate(self, data: List[Tuple[List[str], List[str]]], **kwargs) -> float:
        """
        :param data:
        :param kwargs:
        :return: the accuracy of this model.
        """
        preds = self.decode(data)
        labels = [y for y, _ in data]
        acc = ChunkF1()
        for pred, label in zip(preds, labels):
            acc.update(pred, label)
        return float(acc.get()[1])

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    config = tf.ConfigProto()
    K.tensorflow_backend.set_session(tf.Session(config=config))
    resource_dir = os.environ.get('RESOURCE')
    trn_data = tsv_reader(resource_dir, 'conll03.eng.trn.tsv')
    dev_data = tsv_reader(resource_dir, 'conll03.eng.dev.tsv')
    tst_data = tsv_reader(resource_dir, 'conll03.eng.tst.tsv')
    sentiment_analyzer = NamedEntityRecognizer(resource_dir)
    sentiment_analyzer.train(trn_data, dev_data)
    sentiment_analyzer.evaluate(tst_data)
    sentiment_analyzer.save(os.path.join(resource_dir, 'hw3-model'))
