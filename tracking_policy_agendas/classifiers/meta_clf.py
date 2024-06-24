"""
Meta Class for Classifiers

....................................................................................................
MIT License
Copyright (c) 2021-2023 AUT Iran, Mohammad H Forouhesh
Copyright (c) 2021-2022 MetoData.ai, Mohammad H Forouhesh
....................................................................................................
This module abstracts classifiers.
"""

import os
import pickle
from typing import List, Union

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report

from ..api import get_resources
from ..preprocess.preprocessing import remove_redundant_characters, remove_emoji
from ..word2vec.w2v_emb import W2VEmb


class MetaClf:
    def __init__(self, classifier_instance, embedding_doc: list = None, load_path: str = None):
        self.clf = classifier_instance
        self.emb = W2VEmb()
        self.scaler = None
        self.dir_path = os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.realpath(__file__)))) + "/"
        if load_path is not None:
            get_resources(self.dir_path, resource_name=load_path)
            self.load_model(load_path)
        else:
            self.emb = W2VEmb(embedding_doc)

    def prep_scaler(self, encoded: List[np.ndarray]) -> MinMaxScaler:
        """
        Fitting a Min-Max Scaler to use in the pipeline
        :param encoded:     An array of numbers.
        :return:            A MinMaxScaler
        """
        scaler = MinMaxScaler()
        scaler.fit(encoded)
        return scaler

    def fit(self, X_train, y_train):
        encoded = list(map(self.emb.encode, tqdm(X_train)))
        self.scaler = self.prep_scaler(encoded)
        self.clf.fit(self.scaler.transform(encoded), list(y_train))
        print('============================trian============================')
        print(classification_report(y_train, self.clf.predict(self.scaler.transform(encoded))))
        return self.clf

    def predict(self, X_test, y_test):
        encoded = list(map(self.emb.encode, tqdm(X_test)))
        print('score: ', self.clf.score(self.scaler.transform(encoded), list(y_test)))
        print('=============================test============================')
        print(classification_report(y_test, self.clf.predict(self.scaler.transform(encoded))))

    def load_model(self, load_path: str) -> None:
        """
        A tool to load model from disk.
        :param load_path:   Model path.
        :return:            None
        """

        loading_prep = lambda string: f'model_dir/{load_path}/{string}'
        self.clf.load_model(loading_prep('model.json'))
        self.emb.load(loading_prep('emb.pkl'))
        with open(loading_prep('scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)

    def save_model(self, save_path: str):
        """
        A tool to save model to disk
        :param save_path:   Saving path.
        :return:            None.
        """
        os.makedirs(f'model_dir/{save_path}', exist_ok=True)
        saving_prep = lambda string: f'model_dir/{save_path}/{string}'
        self.clf.save_model(saving_prep('model.json'))
        self.emb.save(saving_prep('emb.pkl'))
        with open(saving_prep('scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f, pickle.HIGHEST_PROTOCOL)

    def __getitem__(self, item: str) -> int:
        """
        getitem overwritten
        :param item:    Input text
        :return:        Predicted class (0, 1).
        """
        return self.vec_predict(item)

    def vec_predict(self, input_text: str) -> int:
        """
        Prediction method.
        :param input_text:  input text, string
        :return:            predicted class. (0, 1)
        """
        prep_text = remove_redundant_characters(remove_emoji(input_text))
        vector = self.scaler.transform(self.emb.encode(prep_text).reshape(1, -1))
        return self.clf.predict(vector)[0]
