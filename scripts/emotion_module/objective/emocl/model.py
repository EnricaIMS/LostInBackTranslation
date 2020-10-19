# standard libraries
import logging
import random
import os
import time

# related third party
import dill
import numpy as np
import tabulate
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torchtext.data import Dataset, Example, Field
from torchtext.data import Iterator
from tqdm import tqdm

# library imports
from cfgparser import global_config
from objective.emocl.ued_handler import UEDDataset, UEDLoader
from objective.emocl.nn.modules import Embed, RNNEncoder
from objective.emocl.nn.models import ModelWrapper
from objective.emotion import EmotionClassifier
from preprocessing import SpacyPreprocessor
from selection.attention import AttentionSelector

# initialize logger
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelTrainer:

    def __init__(self, text_field, label_field, training_dataset, validation_dataset, embeddings=None, **kwargs):

        # TODO: make sure that vocab was build for fields
        parameter_string = "\n".join([key + '=' + str(value) for key, value in kwargs.items()])
        logger.info('New Trainer created with following arguments:\n' + parameter_string)

        self.TEXT = text_field
        self.LABEL = label_field

        if embeddings is not None:
            logger.info('Use pretrained embeddings.')
            self.model = ModelWrapper(
                embeddings=embeddings,
                out_size=len(self.LABEL.vocab), 
                num_embeddings=len(self.TEXT.vocab), 
                **kwargs)
        else:
            logger.info('No pretrained embeddings are used.')
            self.model = ModelWrapper(
                out_size=len(self.LABEL.vocab), 
                num_embeddings=len(self.TEXT.vocab), 
                **kwargs
            )
        logger.info('Model architecture:\n' + str(self.model))

        self.model.to(DEVICE)


        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset

    def get_scores_on_dataset(self, dataset, batch_size):
        iter = Iterator(dataset, batch_size=batch_size, device=DEVICE)

        self.model = self.model.eval()
        gold = []
        pred = []

        with torch.no_grad():
            for data in iter:
                (inputs, inputs_lengths), labels = data.text, data.label
                outputs, _ = self.model(inputs, inputs_lengths)

                _, predicted = torch.max(outputs, 1)
                gold += labels.tolist()
                pred += predicted.tolist()

        assert len(gold) == len(pred)
        accuracy = accuracy_score(gold, pred)
        prec, recall, f1, _ = precision_recall_fscore_support(gold, pred, average=None)

        self.model = self.model.train()

        return accuracy, prec, recall, f1

    def inspect_attention(self, example_sentence):
        prepr = SpacyPreprocessor()
        example_sentence_tokenized = prepr.tokenize(example_sentence)
        emoclass = EmotionClassifier(self.model, self.TEXT, self.LABEL)
        att_sel = AttentionSelector(example_sentence_tokenized, emoclass)
        attentions = att_sel.get_attention_score()
        # put model in training mode again
        self.model = self.model.train()
        return att_sel.return_attention_table(example_sentence_tokenized, attentions)

    def train(self, num_epochs, batch_size):
        logger.info('Start model training with {} iterations.'.format(num_epochs))
        train_iter = Iterator(self.training_dataset, batch_size=batch_size, shuffle=True, device=DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), weight_decay=0.0)
        for epoch in range(num_epochs):
            logger.info('Star training epoch #{}'.format(epoch+1))

            num_batches = 0
            running_loss = 0.0

            with tqdm(total=len(self.training_dataset)) as pbar:
                for i, data in enumerate(train_iter):

                    (inputs, inputs_lengths), labels = data.text, data.label

                    optimizer.zero_grad()

                    outputs, _ = self.model(inputs, inputs_lengths)

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                    num_batches += 1
                    pbar.update(batch_size)

            logger.info('Loss on training: {:.3f}'.format(running_loss/num_batches))
            valid_acc, valid_prec_by_class, valid_rec_by_class, valid_f1_by_class = self.get_scores_on_dataset(
                self.validation_dataset,
                batch_size)

            logger.info('Validation: Total accuracy): {:.3f}'.format(valid_acc))
            table = []
            for i in range(len(self.LABEL.vocab)):
                table.append([
                    self.LABEL.vocab.itos[i], 
                    valid_prec_by_class[i],
                    valid_rec_by_class[i],
                    valid_f1_by_class[i]
                ])
            logger.info("Validation: Results by class:\n" \
                 + tabulate.tabulate(table, headers=["Emotion", "Precision", "Recall", "F1"]))

            logger.info("Attention scores on test sentence:\n" \
                + self.inspect_attention("when I lost someone that was close to me."))

        print('Finished Training')

    def save_model(self, path):
        torch.save(self.model, path)
        logger.info('Model saved to ' + path)

    def save_fields(self, path):
        fields = {}
        for name, field in zip(['TEXT', 'LABEL'], [self.TEXT, self.LABEL]):
            if field.vocab is None:
                logger.warn('Vocab for field {} was not initialized'.format(name))
            fields[name] = field

        with open(path, 'wb') as f:
            dill.dump(fields, f)
        logger.info('Saved fields to ' + path)


class ModelHelper:

    def __init__(self):
        prepr = SpacyPreprocessor()
        self.TEXT = Field(sequential=True, tokenize=prepr.tokenize, lower=True, include_lengths=True, batch_first=True)
        self.LABEL = Field(sequential=False, unk_token=None)

    def initialize_ue_dataset(self, path, split_ratio=(0.9, 0.1), shuffle=True, random_seed=42, **kwargs) -> tuple:
        loader = UEDLoader(path)
        logger.info('Loading UED from {}'.format(path))

        # pass kwargs to filter function
        filtered_dataset = loader.filter_datasets(**kwargs)

        if shuffle:
            random.seed(random_seed)
            random.shuffle(filtered_dataset.examples)

        train, val = filtered_dataset.split(ratio=split_ratio)
        logger.info('Splitted dataset into training and validation with ratio {}'.format(split_ratio))

        start = time.time()
        train = train.to_torchtext(self.TEXT, self.LABEL)
        val = val.to_torchtext(self.TEXT, self.LABEL)
        end = time.time()
        logger.info('Preprocessed training and validation dasets in {:.2f} seconds.'.format(end-start))

        return (train, val)

    def initialize_fields(self, dataset):
        self.TEXT.build_vocab(dataset)
        self.LABEL.build_vocab(dataset)
        logger.info('Initialized vocab ond TEXT and LABEL fields.')

    def load_embeddings(self, embeddings_path, dim):
        if self.TEXT.vocab is None:
            raise RuntimeError('Vocab is not inizialized, run initialize_fields first!')
        stoi = {}
        vectors = []
        embeddings_counter = 0
        with open(embeddings_path, 'r') as f:
                logger.info('Loading embeddings from: {}'.format(embeddings_path))
                start = time.time()
                for i, line in tqdm(enumerate(f)):
                    line_split = line.split()
                    if i == 0 and len(line_split) == 2:
                        dim = int(line_split[1])
                    else:
                        s = line_split[0]
                        vec = torch.tensor([float(d) for d in line_split[1:]], dtype=torch.float)
                        stoi[s] = embeddings_counter
                        vectors.append(vec)
                        embeddings_counter += 1
                end = time.time()
                logger.info('{} embeddings loaded in {:.2f} seconds.'.format(embeddings_counter, end-start))

        self.TEXT.vocab.set_vectors(stoi, vectors, dim)



def make_model_trainer(ued_dataset, params, embeddings=None, embeddings_dim=None): 
    mh = ModelHelper()
    train, val = mh.initialize_ue_dataset(
        path=os.path.join(global_config['directories']['datasets_dir'], 'unified-dataset.jsonl'), 
        source=ued_dataset)
    mh.initialize_fields(train)

    if embeddings is not None:
        embeddings_path = os.path.join(global_config['directories']['embed_dir'], embeddings)
        mh.load_embeddings(embeddings_path, embeddings_dim)
        params['embeddings'] = mh.TEXT.vocab.vectors


    trainer = ModelTrainer(
        text_field=mh.TEXT, 
        label_field=mh.LABEL, 
        training_dataset=train, 
        validation_dataset=val, 
        **params)

    return trainer
