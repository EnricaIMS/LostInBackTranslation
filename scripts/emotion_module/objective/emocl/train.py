import logging
import os
import re
import time

import click

from cfgparser import global_config, parse_config
import objective.emocl.model as model

# initialize logger
emocl_logger = logging.getLogger('objective.emocl')
emocl_logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s - %(message)s')
console_formatter = logging.Formatter('%(levelname)s - %(message)s')
fh = logging.FileHandler(os.path.join(global_config['directories']['log_dir'], 'traing.log'))
sh = logging.StreamHandler()
fh.setFormatter(formatter)
sh.setFormatter(console_formatter)
emocl_logger.addHandler(fh)
emocl_logger.addHandler(sh)

def return_model_parameters(model_section):
    train_params = dict(
        embed_finetune=model_section.getboolean('embed_finetune'),
        embed_noise=float(model_section['embed_noise']),
        embed_dropout=float(model_section['embed_dropout']),
        encoder_dropout=float(model_section['encoder_dropout']),
        encoder_size=int(model_section['encoder_size']),
        encoder_layers=int(model_section['encoder_layers']),
        encoder_bidirectional=model_section.getboolean('encoder_bidirectional'),
        attention=model_section.getboolean('attention'),
        attention_layers=int(model_section['attention_layers']),
        attention_dropout=float(model_section['attention_dropout']),
        attention_context=model_section.getboolean('attention_context'),
        attention_activation=model_section.get('attention_activation', 'tan')
    )
    return train_params

@click.command()
@click.argument('config_path', type=click.Path(exists=True))
def main(config_path):
    config = parse_config(config_path)
    model_section = config['emocl']

    model_name = model_section['name']
    model_filename = model_name + "_emocl.pt"
    fields_filename = model_name + "_fields.dill"

    # check if model dumps for the config section already exist in pretrained dir
    pretrained_dir = global_config['directories']['pretrained_dir']

    if os.path.exists(os.path.join(pretrained_dir, model_filename)):
        emocl_logger.warning("File with name '{}' already exists. Will overwrite model.".format(model_filename))

    embeddings_dim = None
    embeddings = model_section.get('embeddings')
    if embeddings is not None:
        embeddings_dim = int(model_section['embeddings_dim'])

    batch_size = int(model_section['train_batch'])
    num_epochs = int(model_section['num_epochs'])

    train_params = return_model_parameters(model_section)

    trainer = model.make_model_trainer(
        model_section['dataset'], 
        train_params, 
        embeddings=embeddings,
        embeddings_dim=embeddings_dim)

    trainer.train(num_epochs=num_epochs, batch_size=batch_size)

    trainer.save_model(os.path.join(pretrained_dir, model_filename))
    trainer.save_fields(os.path.join(pretrained_dir, fields_filename))

if __name__ == '__main__':
    main()
