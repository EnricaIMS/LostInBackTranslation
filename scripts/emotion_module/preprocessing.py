import spacy
from abc import ABC, abstractmethod

import json


class Dataset(ABC):
    """ Base class for handling emotion datasets """

    @abstractmethod
    def make_iterator(self):
        """ This method should return an iterator over all instances in the dataset """
        pass

    @abstractmethod
    def split(self):
        """ This method should split the given dataset and return new Dataset instances according
        to a pre-defined split ratio """
        pass


class SpacyPreprocessor:

    def  __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def process_input(self, input_sentence):
        return self.nlp(input_sentence.lower())

    def tokenize(self, input_sentence):
        doc = self.process_input(input_sentence)
        print(doc)
        return [token.text for token in doc]
