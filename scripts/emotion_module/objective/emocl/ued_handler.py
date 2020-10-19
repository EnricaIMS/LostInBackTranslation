import logging

from base import Dataset

from torchtext.data import Dataset, Example, Field

from collections.abc import Sequence

import json
import random
from tqdm import tqdm

logger = logging.getLogger(__name__)


class UEDDataset(Dataset):

    def __init__(self, examples):
        assert all(isinstance(ex, UEDExample) for ex in examples), "Argument is not a list of UEDExample objects!"
        self.examples = examples

    def make_iterator(self):
        pass

    def return_random_example(self):
        return random.choice(self.examples)

    def split(self, ratio=(0.8,0.2)):
        try:
            assert type(ratio) is tuple and len(ratio) > 1 and sum(ratio) == 1
        except AssertionError:
            raise ValueError

        splits = []

        num_examples = len(self.examples)
        split_start = 0
        split_end = 0
        for i, fraction in enumerate(ratio):
            num_examples_in_fraction = round(fraction * num_examples)
            if num_examples_in_fraction == 0:
                raise ValueError("Ratio leads to slice with size zero, please adjust.")

            if i == len(ratio)-1:
                splits.append(self.examples[split_start:])
            else:
                split_end += num_examples_in_fraction
                splits.append((self.examples[split_start:split_end]))
                split_start += num_examples_in_fraction

        return [UEDDataset(examples=split) for split in splits]

    def to_torchtext(self, text_field, label_field, lower=True):

        if not isinstance(text_field, Field) or not isinstance(label_field, Field):
            raise TypeError('Arguments are not valid Field instances!')

        tt_examples = []
        print('Building torchtext dataset ')
        for example in tqdm(self.examples):
            try:
                tt_example = Example()
                if lower:
                    text = example.text.lower()
                else:
                    text = example.text
                setattr(tt_example, 'text', text_field.preprocess(text))
                setattr(tt_example, 'label', label_field.preprocess(example.label))
                tt_examples.append(tt_example)
            except AttributeError as err:
                print(err, ", skipping")
                continue
        return Dataset(examples=tt_examples, fields=[('text', text_field), ('label', label_field)])

    def return_emotion_classes(self, include_noemo=True):
        """ Returns the emotion classes that are used in the current datasets as a list """
        dataset_emotions = []
        for ex in self.examples:
            for em, act in ex.props['emotions'].items():
                if act == 1 and em not in dataset_emotions:
                    dataset_emotions.append(em)

        if not include_noemo:
            try:
                dataset_emotions.remove('noemo')
            except ValueError:
                logger.warning('include_noemo is set to False, but noemo category is not present in current dataset')
        
        return dataset_emotions


class UEDLoader:

    def __init__(self, path):
        self.datasets = []
        self.all_examples = []

        with open(path, 'r') as infile:
            for line in infile:
                ex = UEDExample(line)

                dataset = ex.props['source']
                if dataset not in self.datasets:
                    self.datasets.append(dataset)

                if len(ex.props['text']) > 0:
                    self.all_examples.append(ex)

    def filter_datasets(self, enforce_single=True, **kwargs) -> UEDDataset:
        """
        This function returns a subsection of the UED according to keyword arguments matching the respective
        key-value pairs in the UED.

        Values of keyword arguments can either be single strings or a sequence of strings.
        Sequences of strings for a given key are combined with a logical OR.

        If not kwargs are passed, the complete dataset will be returned

        """

        logger.info('Filtering dataset with criteria: {}'.format(', '.join([key + '=' + value for key, value in kwargs.items()])))

        filtered_examples = []

        for example in self.all_examples:
            accepted = True
            for key, value in kwargs.items():
                if isinstance(value, str):
                    if example.props.get(key) != kwargs[key]: accepted = False
                elif isinstance(value, Sequence):
                    if example.props.get(key) not in value: accepted = False
                else:
                    raise TypeError('Invalid argument, must either be string or Sequence of strings!')

                # sometimes, although labeled=single is set, not all returned examples contain truly only
                # one 'activated' emotion. The flag enforce_single double checks and modifies if necessary
                if enforce_single:
                    if not example.is_single_emotion(): 
                        accepted = False
            if accepted:
                filtered_examples.append(example)

        logger.info('Length of resulting dataset: {} instances.'.format(len(filtered_examples)))
        return UEDDataset(filtered_examples)


class UEDExample:

    def __init__(self, json_string):
        self.props = json.loads(json_string)
        self.text = self.props['text']

        if self.is_single_emotion():
            self.label = self.get_single_emotion()

    def is_single_emotion(self):
        """ assure that only ONE emotion is 'active' in the example """
        if sum([value for value in self.props['emotions'].values() if value is not None]) == 1:
            return True
        return False

    def get_single_emotion(self):
        if not self.is_single_emotion():
            activated_emotions = [emotion for emotion, activation in self.props['emotions'].items() if activation == 1]
            raise ValueError('Example with id {} is not single labeled! Activated emotions: {}'.format(
                self.props['id'], len(activated_emotions)
            ))

        for emotion, activation in self.props['emotions'].items():
            if activation == 1:
                return emotion


if __name__ == "__main__":
    loader = UEDLoader('datasets/unified-dataset.jsonl')
    tec = loader.filter_datasets(source='tec')
    print(tec.return_emotion_classes(include_noemo=False))
