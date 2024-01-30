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
from tqdm.notebook import tqdm as tqdm_notebook
from catboost import CatBoostClassifier
from telegram.ext import Updater, CommandHandler, MessageHandler, CallbackContext, filters as Filters
from telegram import Update
import tensorflow as tf
import re
from sklearn.feature_extraction.text import CountVectorizer


def preprocess_txt(line, sw, morpher, exclude, skip_stop_word=False):
    spls = "".join(i for i in line.strip() if i not in exclude).split()
    spls = [morpher.parse(i.lower())[0].normal_form for i in spls]
    spls = [i for i in spls if i != ""]
    if skip_stop_word:
        spls = [i for i in spls if i not in sw]
    return spls


def tqdm_preprocess_txt(kwargs):
    lines = kwargs['lines']
    sw = kwargs['sw']
    morpher = kwargs['morpher']
    exclude = kwargs['exclude']
    skip_stop_word = kwargs['skip_stop_word']

    sentences = []
    for line in tqdm_notebook(lines):
        spls = preprocess_txt(line=line,
                              sw=sw,
                              morpher=morpher,
                              exclude=exclude,
                              skip_stop_word=skip_stop_word)
        sentences.append(spls)
    return sentences


def check_in_list(text, key_list):
    for w in text:
        if w in key_list:
            flag = True
            break
        else:
            flag = False
    return flag


def embed_txt(txt, idfs, midf, modelFT):
    n_ft = 0
    vector_ft = np.zeros(100)
    for word in txt:
        if word in modelFT.wv:
            vector_ft += modelFT.wv[word] * 1
            n_ft += 1
    return vector_ft / n_ft


def do_chat(text, kwargs):
    sw = kwargs['sw']
    morpher = kwargs['morpher']
    exclude = kwargs['exclude']
    modelFT = kwargs['modelFT']
    ft_index = kwargs['ft_index']
    index_map = kwargs['index_map']

    input_txt_cleaned = preprocess_txt(line=text,
                                       sw=sw,
                                       morpher=morpher,
                                       exclude=exclude,
                                       skip_stop_word=True)

    vect_ft = embed_txt(txt=input_txt_cleaned,
                        idfs={},
                        midf=1,
                        modelFT=modelFT)
    ft_index_val, distances = ft_index.get_nns_by_vector(vect_ft, 1, include_distances=True)
    answer = ''
    if distances[0] > 0.45:
        answer = 'Я не понимаю тебя, перефразируй'
    else:
        answer = index_map[ft_index_val[0]]

    return answer


def load_chat_params():
    params_chat = {
        # 'sw': sw,
        # 'morpher': morpher,
        # 'exclude': exclude,
        # 'modelFT': modelFT,
    }

    with open('./params_chat.pkl', 'rb') as f:
        params_chat = pickle.load(f)

    with open('./index_map.pkl', 'rb') as f:
        params_chat['index_map'] = pickle.load(f)

    ft_index = annoy.AnnoyIndex(100, 'angular')
    ft_index.load('./speaker.ann')
    params_chat['ft_index'] = ft_index

    return params_chat
