'''
Tkes a (tab separated) file with the following format:
Sentence id\tEmotion\tTargetEmotions\tText
(The name of the corpus from which the text comes must be in config.ini)

Returns a file in ../../data/ which is just like the input, 
but with Emotion being the predicted emotion.                                 

'''

import torch, configparser, warnings, datetime
import torch.nn as nn
from torch.autograd import Variable
import sys, dill, re

from objective.emotion import EmotionClassifier

script='./scripts/emotion_module/classify_file.py'
warnings.filterwarnings('ignore', '.*Source*')

# Different labels, depending on the used corpus
allLabels={'BLOGS':{"Joy":1, "Fear":2,"Anger":3, "Sadness":4,"Disgust":5, "Noemo":6, "Surprise":0},
           'DIALOGUES':{"Joy":1, "Fear":2,"Anger":3, "Sadness":4,"Disgust":5, "Noemo":6, "Surprise":0 },
           'ISEAR':{"Joy":1, "Fear":2, "Anger":3, "Sadness":4,"Disgust":5, "Shame":6, "Guilt":0 },
           'TALES':{"Joy":1, "Fear":2,"Anger":3, "Sadness":4,"Disgust":5, "Noemo":6, "Surprise":0 },
           'TEC':{"Joy":1, "Fear":2,"Anger":3, "Sadness":4,"Disgust":5, "Surprise":0 }}


def configFile(inifile):
    config = configparser.ConfigParser()
    config.sections()
    config.read(inifile)
    parameters={}
    for section in config.sections():
        for options in config.options(section):
            parameters[section]=config.get(section, options)
    return parameters

class BiLSTM:
    def loadModel(self,currentCorpus:str):
        fields_path = './trained-classifiers/' + 'BiLSTM' + '-' + currentCorpus + "_fields.dill"
        self.model = torch.load('./trained-classifiers/' + 'BiLSTM' + '-' + currentCorpus +".pt")
        self.model.eval()
        with open(fields_path, 'rb') as f:
            self.fields = dill.load(f)
        self.emocl = EmotionClassifier(self.model, self.fields['TEXT'], self.fields['LABEL'])           
    def getPredictions(self,text:str):
        scores = self.emocl.get_scores(text)
        return scores
    
def makePrediction(sentence):
    scores=EC.getPredictions([sentence])[0]
    currentEmotion=max(scores, key=scores.get)
    return(currentEmotion,scores)
            
def classify_file(nameInput,emo_labels):
    f=open('../../data/classified_'+currentCorpus+'.txt','w')
    ordered_emotions=[emo.lower() for emo in sorted(emo_labels, key=emo_labels.get)]
    with open(nameInput, 'r') as myfile:
        for line in myfile:        
            #classify the original sentence
            sentence=line.split('\t')[-1].strip()
            target_emotions=line.split('\t')[-2].strip()
            ids=line.split('\t')[0]
            original_emotion=line.split('\t')[1]
            sentence=re.sub('[^A-Za-z0-9,;:\-\(\)\'\"\!\?\.]',' ', sentence)
            emotions_scores=makePrediction(sentence)
    
            emoscore=emotions_scores[1]

            emoscore=[str(emoscore[emo]) for emo in ordered_emotions]
            scores='\t'.join(emoscore)

            f.write(ids+'\t'+emotions_scores[0]+'\t'+target_emotions+'\t'+sentence+'\n') #emotions_scores[0] is the predicted emotion   
    
    f.close()

if __name__ == "__main__":
    nameInput=sys.argv[1]    
    print("{} : {} : STATUS : Classifying {} .".format(script, datetime.datetime.now(),nameInput))

    # Take name of corpus from which data comes
    parameters=configFile('../../config/config.ini')
    locals().update(parameters)
    currentCorpus=parameters['Data'].strip()

    # Use the model trained on it
    EC=BiLSTM()
    EC.loadModel(currentCorpus)

    # Classify the file
    emo_labels = allLabels[currentCorpus]
    classify_file(nameInput,emo_labels)
    
    print("{} : {} : STATUS : File has been classified in ../../data/classified_{}.txt .".format(script, datetime.datetime.now(),nameInput))
