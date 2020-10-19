import dill
import torch
import logging

import numpy as np
from torchtext import data

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class ClassifierHelper:

    def __init__(self, model, text_field, label_field):
        if not isinstance(model, torch.nn.Module):
            raise TypeError('Argument must be a pytorch Module')

        if not isinstance(text_field, data.Field) or not isinstance(label_field, data.Field):
            raise TypeError('Arguments must be torchtext Field instances.')

        self.model = model.eval().to(DEVICE)
        self.text_field = text_field
        self.label_field = label_field

    @classmethod
    def is_tokenized(clf, input_sentences):
        """ checks whether input is a list of tokenized strings """
        for sentence in input_sentences:
            if not isinstance(sentence, list):
                return False
            if not all(isinstance(token, str) for token in sentence):
                return False
        return True

    def return_input_batch(self, input_sentences, tokenized=False):
        if not any(len(s) > 0 for s in input_sentences):
            raise ValueError('Received empty input sequence')
        if not tokenized:
            tokenized_sentences = []
            for sentence in input_sentences:
                try:
                    tokenized = self.text_field.preprocess(sentence)
                except TypeError as err:
                    logger.warning("Error while preprocessing {}:\n{}".format(sentence, err))
                else:
                    tokenized_sentences.append(tokenized)
        else:
            tokenized_sentences = input_sentences

        return self.text_field.process(tokenized_sentences, device=DEVICE)

    def yield_input_batches(self, input_sentences, tokenized=False, batch_size=None):
        if batch_size is None:
            batch_size = len(input_sentences)

        if tokenized and not self.is_tokenized(input_sentences):
            raise TypeError('"tokenized" set to true, but "input_sentences" is not list of untokenized sentences.')

        if not tokenized:
            # preprocess inputs sentences
            logger.info("Tokenizing input sentences...")
            input_sentences = [self.text_field.preprocess(s) for s in input_sentences]

        for pos in range(0, len(input_sentences), batch_size):
            yield self.text_field.process(input_sentences[pos:pos + batch_size], device=DEVICE)


    def get_outputs(self, batch):
        data, lengths = batch
        with torch.no_grad():
            outputs, attention = self.model(data, lengths)
        return outputs, attention 

    def predict_emotions(self, input_sentences):
        predictions = []
        for batch in self.yield_inputs(input_sentences):
            _, pred = torch.max(self.get_outputs(batch), 1)

            predictions += pred.tolist()

        return [self.label_field.vocab.itos[pred] for pred in predictions]

    def get_probabilites(self, outputs):
        def softmax(x):
            return np.exp(x)/np.sum(np.exp(x), axis=0)

        if len(outputs.shape) == 1:
            outputs = outputs.view(1, -1)

        probs = []
        for output in outputs:
            probs.append(softmax(output.cpu().numpy()))
        return probs

    def get_probabilites_by_class(self, outputs):
        probs = self.get_probabilites(outputs)

        probs_by_class = []
        # probs is a list of numpy arrays
        for p in probs:
            prob_by_class = {self.label_field.vocab.itos[i]: p[i] for i in range(len(self.label_field.vocab))}
            probs_by_class.append(prob_by_class)
        return probs_by_class


class EmotionClassifier(ClassifierHelper):

    def __init__(self, model, text_field, label_field):
        super(EmotionClassifier, self).__init__(model, text_field, label_field)

    def get_scores(self, input_sentences):
        
        if len(input_sentences) > 0:
            tokenized = self.is_tokenized(input_sentences)

            probs = []

            for batch in self.yield_input_batches(input_sentences, tokenized=tokenized, batch_size=128):
                outputs, _ = self.get_outputs(batch)
                probs += self.get_probabilites(outputs)
               
            sentence_scores = []
            
            for p in probs:
                scores = {self.label_field.vocab.itos[emotion_index]: score for emotion_index, score in enumerate(p)}
                sentence_scores.append(scores)

            return sentence_scores
        else:
            raise ValueError('Received empty input sequence')

    def get_score_for_target_emotion(self, input_sentences, target_emotion):
        if len(input_sentences) > 0:
            tokenized = self.is_tokenized(input_sentences)

            probs = []

            for batch in self.yield_input_batches(input_sentences, tokenized=tokenized, batch_size=128):
                outputs, _ = self.get_outputs(batch)
                probs += self.get_probabilites(outputs)

            sentence_scores = []

            target_emotion_index = self.label_field.vocab.stoi[target_emotion]

            for p in probs:
                score = {'emotion': p[target_emotion_index]}
                sentence_scores.append(score)

            return sentence_scores
        else:
            raise ValueError('Received empty input sequence')

    def inspect_attention(self, input_sentence, emoclass):
        print(sentence)
        print(emoclass)
        prepr = SpacyPreprocessor()
        example_sentence_tokenized = prepr.tokenize(input_sentence)
        att_sel = AttentionSelector(example_sentence_tokenized, emoclass)
        attentions = att_sel.get_attention_score()
        print(attention)
 
        return att_sel.return_attention_table(example_sentence_tokenized, attentions)



if __name__ == '__main__':

    model = torch.load('emoclass.pt')

    with open('fields.dill', 'rb') as f:
        fields = dill.load(f)

    sentences = []
    with open('/home/dave/Development/Python/emotion-transfer/emotion-transfer/book.txt') as f:
        for line in f:
            if not line.startswith('#'):
                sentences.append(line.strip())

    ec = EmotionClassifier(model, fields['text'], fields['label'])
    score_list = ec.get_scores(sentences)

    for sentence_score in score_list.return_emotion_top_k('joy'):
        if 'love' in sentence_score.sentence:
            print(sentence_score.sentence, ' - ', sentence_score.scores['joy'])
