'''
Takes a (tab separated) file with the following format:
Sentence id\tEmotion\tText
(The name of the corpus from which the text comes must be in config.ini)

Returns a file in ../../data/ of the form Sentence id\tEmotion\tTargetEmotion\tText
where Emotion is now the predicted emotion, and the target emotion depends on the
goal of pipeline usage.                    

'''

import torch, configparser, warnings, datetime
import sys, dill, re
import torch.nn as nn
import pandas as pd

from torch.autograd import Variable
sys.path.append('./emotion_module')
from emotion_module.objective.emotion import EmotionClassifier

script='./scripts/emotion_module/classify_file.py'
warnings.filterwarnings('ignore', '.*Source*')

# Different labels, depending on the used corpus
def chooseEmoLabels(corpus):
    allLabels={'BLOGS':{"Joy":1, "Fear":2,"Anger":3, "Sadness":4,"Disgust":5, "Noemo":6, "Surprise":0},
            'DIALOGUES':{"Joy":1, "Fear":2,"Anger":3, "Sadness":4,"Disgust":5, "Noemo":6, "Surprise":0 },
            'ISEAR':{"Joy":1, "Fear":2, "Anger":3, "Sadness":4,"Disgust":5, "Shame":6, "Guilt":0 },
            'TALES':{"Joy":1, "Fear":2,"Anger":3, "Sadness":4,"Disgust":5, "Noemo":6, "Surprise":0 },
            'TEC':{"Joy":1, "Fear":2,"Anger":3, "Sadness":4,"Disgust":5, "Surprise":0 }}
    emo_labels = allLabels[corpus]
    return emo_labels

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
    def loadModel(self,currentCorpus:str,path_to_Classifiers):
        fields_path = path_to_Classifiers + 'BiLSTM' + '-' + currentCorpus + "_fields.dill"
        self.model = torch.load(path_to_Classifiers + 'BiLSTM' + '-' + currentCorpus +".pt")
        self.model.eval()
        with open(fields_path, 'rb') as f:
            self.fields = dill.load(f)
        self.emocl = EmotionClassifier(self.model, self.fields['TEXT'], self.fields['LABEL'])           
    def getPredictions(self,text:str):
        scores = self.emocl.get_scores(text)
        return scores
    
def makePrediction(sentence,EC,emoLabels):
    scores=EC.getPredictions([sentence])[0]
    currentEmotion=max(scores, key=scores.get)
    
    ordered_emotions=[emo.lower() for emo in sorted(emoLabels, key=emoLabels.get)]
    scores=[scores[emo] for emo in ordered_emotions]
    return(currentEmotion,scores)

def classify_file(nameInput,currentCorpus,EC):
    f=open('../data/classified_input.txt','w')
    emo_labels=chooseEmoLabels(currentCorpus)
    
    with open(nameInput,'r') as myFile:
        line_number=1    
        # What will be inserted in TargetEmotion column in output file
        # even if the goal is not emotion transfer (RQ3) 
        # -for consistency in file formats among RQ1, RQ2 and RQ3
        target_emotions = ', '.join(emo_labels).lower()

        for line in myFile:

            ids=line_number
            sentence=line.strip()
            sentence=re.sub('[^A-Za-z0-9,;:\-\(\)\'\"\!\?\.]',' ', sentence)
            
            #classify sentence
            emotions_scores=makePrediction(sentence,EC,emo_labels)

            f.write(str(ids)+'\t'+emotions_scores[0]+'\t'+target_emotions+'\t'+sentence+'\n') #emotions_scores[0] is the predicted emotion       
            line_number+=1
    f.close()
    print("{} : {} : STATUS : File has been preprocessed and classified. Output saved in ../data/ .".format(script, datetime.datetime.now(),nameInput))

if __name__ == "__main__":
    nameInput = sys.argv[1]
    path_to_Classifiers = './emotion_module/trained-classifiers/'
    print("{} : {} : STATUS : Classifying {} .".format(script, datetime.datetime.now(),nameInput))

    # Take name of corpus from which data comes and the goal of the analysis
    parameters=configFile('../config/config.ini')
    locals().update(parameters)
    currentCorpus=parameters['Data'].strip()
    goal=parameters['Goal'].strip()

    # Use the model trained on it
    EC=BiLSTM()
    EC.loadModel(currentCorpus,path_to_Classifiers)

    # Classify the file
    classify_file(nameInput,currentCorpus,EC)
