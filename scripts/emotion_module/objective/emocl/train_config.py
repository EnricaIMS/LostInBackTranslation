

ued_path = '/home/dave/Development/Python/emotion-transfer/emotion-transfer/datasets/unified-dataset.jsonl'

# embeddings
use_embeddings = True
embeddings_path = '/home/dave/Development/Python/emotion-transfer/emotion-transfer/objective/emocl/nn/embeddings/ntua_twitter_affect_310.txt'
embeddings_dim = 310

logfile = 'logs/train.log'

filters = {
    'source': 'tec'
}

training_parameters = {
    'embed_finetune':True,
    'embed_noise':0.2,
    'embed_dropout':0.1,
    'encoder_dropout':0.3,
    'encoder_size':250,
    'encoder_layers':2,
    'encoder_bidirectional':True,
    'attention':True,
    'attention_layers':2,
    'attention_dropout':0.3
}

num_epochs = 2
batch_size = 32

output_name = 'emoclass_emoint.pt'
fields_name = 'fields_emoint.dill'