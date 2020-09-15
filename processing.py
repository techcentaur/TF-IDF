import os
import re
import sys
import copy
import nltk
import math
import string
import pickle
import numpy as np
import glob

from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from num2words import num2words

class TextClean:
    def lower_case(self, data):
        return [tok.lower() for tok in data]

    def remove_stop_words(self, data):
        stop_words = stopwords.words('english')

        valid_words = []
        for word in data:
            if word not in stop_words and word != "":
                valid_words.append(word)

        return valid_words

    def apostrophe_normalisation(self, data):        
        for (i, j) in [(r"n\'t", " not"), (r"\'re", " are"), (r"\'s", " is"), (r"\'d", " would"), (r"\'ll", " will"), (r"\'t", " not"), (r"\'ve", " have"), (r"\'m", " am"), (r"\'Cause", "because")]:
            data = re.sub(i, j, data)
        return data

    def punctuation_removal(self, data):
        punc_list = list(punctuation)
        return [i for i in data if i not in punc_list]
        
    def stem_processing(self, data):
        stemmer= PorterStemmer() 
        valid_words = [stemmer.stem(t) for t in data]
        return valid_words

    def convert_numbers_to_string(self, data):
        valid_words = []
        for t in data:
            try:
                t = num2words(int(t))
            except:
                pass
            valid_words.append(t)
        return valid_words

    def cleanse_data(self, str_data):
        str_data = self.apostrophe_normalisation(str_data)
        
        data = word_tokenize(str_data)
        
        data = self.lower_case(data)
        data = self.punctuation_removal(data)
        data = self.convert_numbers_to_string(data)

        # data = self.stem_processing(data)
        data = self.remove_stop_words(data)

        return data

class Processing:
    def get_all_text_files(self, directory):
        return glob.glob(os.path.join(directory, "*"))

    def get_preprocessed_text(self, text_files):
        processed_data = []
        clean = TextClean()

        correct_files = []
        for file in text_files:
            with open(file, 'r', encoding='utf-8') as f:
                try:
                    data = f.read().strip()
                    if data:
                        processed_data.append(clean.cleanse_data(data))
                    correct_files.append(file)
                except:
                    pass
        return processed_data, correct_files


