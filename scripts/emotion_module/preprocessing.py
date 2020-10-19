import spacy

class SpacyPreprocessor:

    def  __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def process_input(self, input_sentence):
        return self.nlp(input_sentence.lower())

    def tokenize(self, input_sentence):
        doc = self.process_input(input_sentence)
        print(doc)
        return [token.text for token in doc]
