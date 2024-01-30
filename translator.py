import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pandas as pd
import string
from datetime import datetime
import pickle
import annoy
from tools import paralell_execution, ar_split_eq_cpu
from multiprocessing import cpu_count
from pymorphy2 import MorphAnalyzer
from stop_words import get_stop_words
from gensim.models import Word2Vec, FastText
import re
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=False,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return state

    def initialize_hidden_state(self):
        # создаем тензор из нулей размера (батч, кол-во ячеек)
        return tf.zeros((self.batch_sz, self.enc_units))


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, x, hidden):
        # x shape после прохождения через эмбеддинг == (batch_size, 1, embedding_dim)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # отправляем в GRU входные данные и скрытое состояние (от энкодера)
        # выход GRU (batch_size, timesteps, units)
        # размер возвращаемого внутреннего состояния (batch_size, units)
        output, state = self.gru(x, initial_state=hidden)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state


def preprocess_sentence(w):  # функция препроцессинга
    # переводим предложение к нижнему регистру и удалем начальные и конечные пробелы
    w = w.lower().strip()

    # отделяем пробелом слово и следующую за ним пунктуацию
    # пример: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # все, кроме букв и знаков пунктуации, заменяем пробелом
    w = re.sub(r"[^a-zA-Zа-яА-Я?.!,']+", " ", w)

    # удаляем лишние пробелы в начале и конце
    w = w.strip()

    # создаем начало и конец последовательности
    # теперь модель знает, где начинать и заканчивать предсказания
    w = '<start> ' + w + ' <end>'
    return w


def evaluate(sentence, params_translator):
    max_length_targ = params_translator['max_length_targ']
    max_length_inp = params_translator['max_length_inp']
    inp_lang = params_translator['inp_lang']
    targ_lang = params_translator['targ_lang']
    units = params_translator['units']
    vocab_inp_size = params_translator['vocab_inp_size']
    vocab_tar_size = params_translator['vocab_tar_size']
    embedding_dim = params_translator['embedding_dim']
    BATCH_SIZE = params_translator['BATCH_SIZE']
    encoder = params_translator['encoder']
    decoder = params_translator['decoder']

    # выполняем препоцессинг предложения
    attention_plot = np.zeros((max_length_targ, max_length_inp))
    # разбиваем предложение по пробелам и составляем список индексов каждого слова
    sentence = preprocess_sentence(sentence)

    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    # заполняем inputs нулями справа до максимальной длины входного текста
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_length_inp,
                                                           padding='post')
    # преобразуем inputs в тензор
    inputs = tf.convert_to_tensor(inputs)

    result = ''
    # инициализируем входной hidden из нулей размера (1, units)
    hidden = [tf.zeros((1, units))]
    # подаем inputs и hidden в encoder
    enc_hidden = encoder(inputs, hidden)

    # инициализируем входной hidden декодера -- выходной hidden энкодера
    dec_hidden = enc_hidden
    # вход декодера -- список [индекс start] размера(1,1)
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

    for t in range(max_length_targ):
        # получаем выход декодера
        predictions, dec_hidden = decoder(dec_input, dec_hidden)

        # storing the attention weights to plot later on
        predicted_id = tf.argmax(predictions[0]).numpy()
        new_trans = targ_lang.index_word[predicted_id]

        # заканчиваем на токене end
        if new_trans == '<end>':
            return result, sentence
        else:
            result += new_trans + ' '

        # предсказанный predicted ID подаем обратно в декодер (размер (1,1))
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence


def translate(text, params_translator):
    start_phrases = ['переведи', 'перевод']
    splt = text.split(' ')
    if splt[0] in start_phrases:
        text = ' '.join(splt[1:])

    max_length_targ = params_translator['max_length_targ']
    max_length_inp = params_translator['max_length_inp']
    inp_lang = params_translator['inp_lang']
    targ_lang = params_translator['targ_lang']
    units = params_translator['units']
    vocab_inp_size = params_translator['vocab_inp_size']
    vocab_tar_size = params_translator['vocab_tar_size']
    embedding_dim = params_translator['embedding_dim']
    BATCH_SIZE = params_translator['BATCH_SIZE']

    translated, inputed_text = evaluate(text, params_translator)
    return translated


def test_translate():
    params_translator = {}
    with open('./params_translator.pkl', 'rb') as f:
        params_translator = pickle.load(f)

    encoder = Encoder(params_translator['vocab_inp_size'],
                      params_translator['embedding_dim'],
                      params_translator['units'],
                      params_translator['BATCH_SIZE'])
    decoder = Decoder(params_translator['vocab_tar_size'],
                      params_translator['embedding_dim'],
                      params_translator['units'],
                      params_translator['BATCH_SIZE'])
    params_translator['optimizer'] = tf.keras.optimizers.Adam()
    params_translator['encoder'] = encoder
    params_translator['decoder'] = decoder

    checkpoint_dir = './training_nmt_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        optimizer=tf.keras.optimizers.Adam(),
        encoder=encoder,
        decoder=decoder,
    )
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    print(translate('переведи добрый день', params_translator))
    print(translate('переведи хороший выбор', params_translator))