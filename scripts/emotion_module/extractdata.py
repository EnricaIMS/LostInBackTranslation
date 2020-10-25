from emotion_module.objective.emocl.ued_handler import UEDDataset, UEDLoader 
import random, sys, datetime

script='./emotion_module/extractdata.py'

datasets=['isear','tec','blogs','tales']
loader = UEDLoader('/home/users2/troianea/Projekte/projectdata/emotion_classification/LexicalSubstitution4StyleTransfer/David/emotion-transfer/datasets/unified-dataset.jsonl')
   
for name in datasets:
    print("{} : {} : STATUS : Extracting {} data from ../data/unified-corpora".format(script, datetime.datetime.now(),name))

    corpus=name.strip()
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
