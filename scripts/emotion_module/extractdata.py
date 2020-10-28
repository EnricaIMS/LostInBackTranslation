'''
Extracts text used in our experiment in folder data.
'''

from emotion_module.objective.emocl.ued_handler import UEDDataset, UEDLoader 
import random, sys, datetime

script='./emotion_module/extractdata.py'

datasets=['ISEAR','TEC','BLOGS','TALES']
loader = UEDLoader('../data/unified-corpora/unified-dataset.jsonl')
   
for name in datasets:
    print("{} : {} : STATUS : Extracting {} data from ../data/unified-corpora".format(script, datetime.datetime.now(),name))

    corpus=name.strip().lower()
    if corpus=='blogs':
        corpus='emotiondata-aman'
    if corpus=='tales':
        corpus='tales-emotion'
    

    filtered_dataset = loader.filter_datasets(source=corpus)

    random_seed=42
    random.seed(random_seed)
    random.shuffle(filtered_dataset.examples)

    train, val, test = filtered_dataset.split(ratio=(0.7,0.1,0.2))
    
    with open('../data/'+name+'.txt','w') as mytest:
        for i in range(len(test)):
            mytest.write(test[i].text+'\n')
