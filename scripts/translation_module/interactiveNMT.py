#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate raw text with a trained model. Batches data on-the-fly.

This allows to call the models from a customized path..
"""

import unicodedata
from collections import namedtuple

import sys
import torch

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data import encoders

Batch = namedtuple('Batch', 'ids src_tokens src_lengths')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')


class iTranslate(object):
    def __init__(self, args):
        sys.argv[1:] = args.split() 
        parser = options.get_generation_parser(interactive=True)
        args = options.parse_args_and_arch(parser)

        utils.import_user_module(args)

        if args.buffer_size < 1:
            args.buffer_size = 1
        if args.max_tokens is None and args.max_sentences is None:
            args.max_sentences = 1

        assert not args.sampling or args.nbest == args.beam, \
            '--sampling requires --nbest to be equal to --beam'
        assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
            '--max-sentences/--batch-size cannot be larger than --buffer-size'

        #print(args)

        self.use_cuda = torch.cuda.is_available() and not args.cpu

        # Setup task, e.g., translation
        self.task = tasks.setup_task(args)

        # Load ensemble
        print('| loading model(s) from {}'.format(args.path))
        self.models, _model_args = checkpoint_utils.load_model_ensemble(
            args.path.split(':'),
            arg_overrides=eval(args.model_overrides),
            task=self.task,
        )

        # Set dictionaries
        self.src_dict = self.task.source_dictionary
        self.tgt_dict = self.task.target_dictionary

        # Optimize ensemble for generation
        for model in self.models:
            model.make_generation_fast_(
                beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
                need_attn=args.print_alignment,
            )
            if args.fp16:
                model.half()
            if self.use_cuda:
                model.cuda()

        # Initialize generator
        self.generator = self.task.build_generator(args)

        # Handle tokenization and BPE
        self.tokenizer = encoders.build_tokenizer(args)
        self.bpe = encoders.build_bpe(args)

        # Load alignment dictionary for unknown word replacement
        # (None if no unknown word replacement, empty if no path to align dictionary)
        self.align_dict = utils.load_align_dict(args.replace_unk)

        self.max_positions = utils.resolve_max_positions(
            self.task.max_positions(),
            *[model.max_positions() for model in self.models]
        )

        self.args = args

    def encode_fn(self, x):
        if self.tokenizer is not None:
            x = self.tokenizer.encode(x)
        if self.bpe is not None:
            x = self.bpe.encode(x)
        return x

    def decode_fn(self, x):
        if self.bpe is not None:
            x = self.bpe.decode(x)
        if self.tokenizer is not None:
            x = self.tokenizer.decode(x)
        return x

    def make_batches(self, lines):
        tokens = [
            self.task.source_dictionary.encode_line(
                self.encode_fn(src_str), add_if_not_exist=False
            ).long()
            for src_str in lines
            ]
        lengths = torch.LongTensor([t.numel() for t in tokens])
        itr = self.task.get_batch_iterator(
            dataset=self.task.build_dataset_for_inference(tokens, lengths),
            max_tokens=self.args.max_tokens,
            max_sentences=self.args.max_sentences,
            max_positions=self.max_positions,
        ).next_epoch_itr(shuffle=False)
        for batch in itr:
            yield Batch(
                ids=batch['id'],
                src_tokens=batch['net_input']['src_tokens'], src_lengths=batch['net_input']['src_lengths'],
            )

    def translate_batch(self, inputs):
        start_id = 0

        results = []
        for batch in self.make_batches(inputs):#, task, max_positions, encode_fn):
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            if self.use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()

            sample = {
                'net_input': {
                    'src_tokens': src_tokens,
                    'src_lengths': src_lengths,
                },
            }
            translations = self.task.inference_step(self.generator, self.models, sample)
            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], self.tgt_dict.pad())
                results.append((start_id + id, src_tokens_i, hypos))

        # sort output to match input order
        hypotheses = []
        for id, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
            if self.src_dict is not None:
                src_str = self.src_dict.string(src_tokens, self.args.remove_bpe)

            # Process top predictions
            for hypo in hypos[:min(len(hypos), self.args.nbest)]:
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'],
                    align_dict=self.align_dict,
                    tgt_dict=self.tgt_dict,
                    remove_bpe=self.args.remove_bpe,
                )
                hypo_str = self.decode_fn(hypo_str)
                hypotheses.append(hypo_str)
                #print('H-{}\t{}\t{}'.format(id, hypo['score'], hypo_str))

        # update running id counter
        start_id += len(inputs)
        sys.stdout.flush()
        return hypotheses


def main():
    args = '--path translation_models/wmt19.en-de.joined-dict.ensemble/model1.pt:translation_models/wmt19.en-de.joined-dict.ensemble/model2.pt:translation_models/wmt19.en-de.joined-dict.ensemble/model3.pt translation_models/wmt19.en-de.joined-dict.ensemble/ --beam 5 --source-lang en --target-lang de --remove-bpe --batch-size 32 --buffer-size 32 --replace-unk --bpe fastbpe --bpe-codes translation_models/wmt19.en-de.joined-dict.ensemble/bpecodes --nbest 5'
    inputs = '''
            Emotions are nice little creatures, but they are lost in translation.
            Participants to the ISEAR experiment described sad events.
            '''.strip().split('\n')
    it = iTranslate(args)
    print(it.translate_batch(inputs))

if __name__ == '__main__':
    main()
