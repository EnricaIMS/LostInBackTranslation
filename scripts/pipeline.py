#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
This script:
    - gives emotion scores to input file.
    - produces backtranslations, scores them wrt emotions.
    - returns the best backtranslation, depending on task
      (RQ1 vs. RQ2 vs. RQ3)
'''
import configparser, sys, argparse
import datetime, re, os, copy

#to access elements of Tensor
from torch.autograd import Variable
from itertools import repeat

#Import the scoring modules
import emotion_module.classify_emotions as css
import translation_module.translate as trans

import numpy as np

version = 0.1
script= './emotion-transfer.py'

path_to_emotionClassifiers='./emotion_module/trained-classifiers/' 
path_to_Output='../output/'


parser = argparse.ArgumentParser()
parser.add_argument("-i","--input", 
                    help="name of txt file to be processed (optional)")
args = parser.parse_args()


#Emo mapping (this will be useful to select the correct emotion score)
#Order of emotion in the classifier output vector
#(will be useful to select the correct emotion score)

emo_maps={'joy': 1, 'fear':2, 'anger':3, 'sadness':4, 'disgust':5}
if 'ISEAR' in args.input:
    emo_maps['guilt']= 0
    emo_maps['shame']=6
if 'TEC' in args.input or 'TALES' in args.input or 'BLOGS' in args.input or 'DIALOGUES' in args.input:
    emo_maps['surprise']= 0
if 'TALES' in args.input or 'BLOGS' in args.input or 'DIALOGUES' in args.input:
    emo_maps['noemo']=6



class Paraphrase(object):    
    '''
    Paraphrase==backtranslation.
    '''
    def __init__(self,sentence="",score=0,EmoScore=0,allEmoScores=0,sourceLanguage="",targetLanguage="",targetEmotion="",sentenceID=""):
        self._sentence=sentence
        self._score=score
        self._EmoScore=EmoScore
        self._allEmoScores=allEmoScores
        self._sourceLanguage=sourceLanguage
        self._targetLanguage=targetLanguage
        self._targetEmotion=targetEmotion
        self._sentenceID=sentenceID
        
def configFile(inifile):
    '''
    The config file sets parameters and
    goals of the pipeline.
    '''
    config = configparser.ConfigParser()
    config.sections()
    config.read(inifile)
    parameters={}

    for section in config.sections():
        for options in config.options(section):
            parameters[section]=config.get(section, options)
    return parameters



def loadTranslationModels(sourceLanguage:str,targetLanguages:list,decoding:str,numbForwtransl:int,numbBacktransl:int):
    '''
    Load the models used in translation.
    '''
    print('{} : {} : STATUS : Loading Translation models.'.format(script, datetime.datetime.now()))

    #Load Translation Model     
    global TRM
    TRM=trans.Translation() #this will load models for all source-target pairs
    TRM.loadTranslationModel(sourceLanguage,targetLanguages,decoding,numbForwtransl,numbBacktransl)      


def scoreInput(originalEmo,sentenceID, sentence:str,trainingData,emoLabels):
    '''
    Give emotion scores to input sentence.
    '''
    sentence=re.sub('[^A-Za-z0-9,;:\-\(\)\'\"\!\?\.]',' ', sentence)
    
    #Load Emotion Classifier   
    global EC    
    try:
        EC
    except NameError:
        print('{} : {} : STATUS : Loading Emotion Model from {}.'.format(script, datetime.datetime.now(), path_to_emotionClassifiers))
        EC=css.BiLSTM()
        EC.loadModel(trainingData,path_to_emotionClassifiers)

    # Scored input will be saved here
    nameFILE=path_to_Output+'scored_input.txt'
    with open(nameFILE,'a') as scoredINPUT:
        labelAndScores=css.makePrediction(sentence,EC,emoLabels) # (classified emo, [emo1score, emo2score...})
        scores=list(labelAndScores[1])
        input_emotion_score=np.max(scores)
        ORIGINALscores=[str(round(x,3)) for x in scores]
        mystr='\t'+'\t'.join(ORIGINALscores)
        scoredINPUT.write(str(sentenceID)+'\t'+originalEmo+'\t'+sentence+mystr+'\n')

    print('{} : {} : STATUS : Processed input sentence: {}.'.format(script, datetime.datetime.now(), str(sentenceID)))
    return(input_emotion_score,ORIGINALscores)


def paraphraseSentences(sentence: str, sentenceID, numbForwtransl:int, numbBacktransl:int, sourceLanguage:str, targetLanguages:list):     
    '''
    Translate forward and back.
    '''
    sentence=re.sub('[^A-Za-z0-9,;:\-\(\)\'\"\!\?\.]',' ', sentence)
    outputsentences=list()
    translations=TRM.forwardbackTranslation(sentence, numbForwtransl, numbBacktransl, sourceLanguage,targetLanguages)

    for fbt in translations:
        instance=Paraphrase(fbt) 
        instance._sentence=fbt[0]
        instance._sourceLanguage=sourceLanguage
        instance._targetLanguage=fbt[1]
        instance._sentenceID=sentenceID
        outputsentences.append(instance)
    
    return(outputsentences)    


def scoreParaphrase(paraphrases:list,goal:str,emoLabels):
    '''
    Emotion Classifier module in pipeline.
    Give emotion scores to backtranslations.
    '''
    scoredParaphrases=list()

    for instance in paraphrases:
        p=instance._sentence
        emoscore=css.makePrediction(p,EC,emoLabels)
        scores=[round(x,3) for x in list(emoscore[1])]
        instance._EmoScore=scores                        
        scoredParaphrases.append(instance)      

    return scoredParaphrases


def get_best_paraphrases(scoredParaphrases:list, numTopParaphrases:int,targetEmotion:list,originalEmo,originalEmoscore,ORIGINALemoscores,goal):
    '''    
    Emotion Informed Selection module in pipeline. 
    Returns best paraphrases, depending on goal of pipeline usage.
    '''
    selected=list()

    for emo in targetEmotion:
        sort_paraphrases=list()
        emo=emo.strip()
        
        if goal=='RQ2' and emo!=originalEmo:
            continue
        
        for p in scoredParaphrases:
            score=0
            p._score=score 
            scores=p._EmoScore
            index_of_target=emo_maps[emo]    
            current_emo=scores[index_of_target] #score of target in paraphrases
        
            # I cannot modify p._Emoscore or I lose info about different emotions
            instance=Paraphrase(p) 
            instance._sentence=p._sentence
            instance._sourceLanguage=p._sourceLanguage
            instance._targetLanguage=p._targetLanguage
            instance._sentenceID=p._sentenceID
            instance._targetEmotion=emo.strip() 
            instance._EmoScore=current_emo

            if goal=='RQ2':
                instance._allEmoScores=scores    
                target_in_original=float(ORIGINALemoscores[index_of_target]) #score of target emo in input
                delta=current_emo-target_in_original
                instance._score+=delta
            else:
                instance._score+=current_emo
            sort_paraphrases.append(instance)
        
        '''
        Rerank, depending on RQ.
        '''
        if goal=='RQ2':
            #take paraphrases with minimum delta with input emotion
            sort_paraphrases = sorted(sort_paraphrases, key=lambda x: x._score)
            deltas = [x._score for x in sort_paraphrases]        
            min_delta=min(deltas, key=lambda x:abs(x-0))
            top_scoring=[sp for sp in sort_paraphrases if sp._score==min_delta][:numTopParaphrases]

        else:
            sort_paraphrases = sorted(sort_paraphrases, key=lambda x: x._score, reverse=True)
            top_scoring=sort_paraphrases[:numTopParaphrases]

        for ts in top_scoring:
            selected.append(ts)


    if goal=='RQ2':
        #take only the paraphrase which minimizes the original emotion
        delta_minimizer=selected[0] #item with most similar emotion to input
        selected=list()
        for emo in targetEmotion:
            #take it many times, i.e. one per target emo
            emo=emo.strip()
            new_paraphrase=copy.deepcopy(delta_minimizer)
            new_paraphrase._targetEmotion=emo #only difference btw one and another
            index_of_target=emo_maps[emo]
            #from all scores of the sentence, take that of current emo
            current_emo=delta_minimizer._allEmoScores[index_of_target]
            new_paraphrase._EmoScore=current_emo
            selected.append(new_paraphrase)
        
    return selected


def processLine(originalEmo,targetEmotion:list, text: str, sentenceID, sourceLanguage: str, targetLanguages:list, numbForwtransl:int, numbBacktransl: int, trainingData:str, numTopParaphrases: int, goal:str,emoLabels):        
    '''
    Get paraphrases/backtranslations for
    every input line.
    '''
    original_emo_score=scoreInput(originalEmo,sentenceID, text, trainingData, emoLabels)
    paraphrases = paraphraseSentences(text, sentenceID, numbForwtransl, numbBacktransl, sourceLanguage, targetLanguages)
    scoredParaphrases=scoreParaphrase(paraphrases,goal,emoLabels)
    
    paraphraseStrings = get_best_paraphrases(scoredParaphrases, numTopParaphrases,targetEmotion,originalEmo, original_emo_score[0],original_emo_score[1],goal)
    
    return paraphraseStrings


def readInput():
    '''
    Process line by line.
    '''
    #if input is file
    if args.input:
        textfilename=args.input
        with open(textfilename) as myf:
            for line in myf.readlines():
                yield line


def translateAndscore(sourceLanguage:str, targetLanguages:str, numbForwtransl:int, numbBacktransl:int, trainingData:str, numTopParaphrases: int, decoding:str, goal: str, emoLabels):
    '''
    Combine all the functions above.
    '''

    resultingParaphrases = list()          
    myinput=readInput()
    
    # Save output here
    outputFile=open(path_to_Output+'scored_backtranslations.txt','w')
        
    for line in myinput:
        line=line.strip().split('\t')                  
        sentenceID=line[0]
        targetEmotion=line[-2].lower().split(',') #may contain more than one target emotion
        sentence=line[-1]
        sentence=re.sub('\ +', ' ',sentence)
        originalEmo=line[1]
        outputsentence=processLine(originalEmo,targetEmotion,sentence,sentenceID, sourceLanguage,targetLanguages, numbForwtransl, numbBacktransl, trainingData, numTopParaphrases, goal, emoLabels)
        resultingParaphrases.append(outputsentence)
        
        for o in outputsentence:
            overall_score=str(round(o._score,3))
            em_score=str(round(o._EmoScore,3))
            outputFile.write(sentenceID+'\t'+o._targetEmotion+'\t'+o._sentence+'\t'+em_score+'\n')
            
    outputFile.close()   
        
    return resultingParaphrases


''' MAIN METHOD '''
if __name__ == "__main__":    
    
    parameters=configFile('../config/config.ini')
    locals().update(parameters)
    numbBacktransl=int(parameters['backtranslation'])
    numbForwtransl=int(parameters['forwardtranslations'])
    numTopParaphrases=int(parameters['n_best'])
    decoding=parameters['Decoding']
    sourceLanguage=parameters['Language'] 
    targetLanguages=parameters['TargetLanguages']
    targetLanguages=[x.strip() for x in targetLanguages.split(',')]
    trainingData=parameters['Data']
    goal=parameters['Goal']

    loadTranslationModels(sourceLanguage,targetLanguages,decoding,numbForwtransl,numbBacktransl)
    

    #Emo mapping (this will be useful to select the correct emotion score)
    #Order of emotion in the classifier output vector
    #(will be useful to select the correct emotion score)

    emo_maps={'joy': 1, 'fear':2, 'anger':3, 'sadness':4, 'disgust':5}
    if 'ISEAR' in trainingData:
        emo_maps['guilt']= 0
        emo_maps['shame']=6
    if 'TEC' in trainingData or 'TALES' in trainingData or 'BLOGS' in trainingData:
        emo_maps['surprise']= 0
    if 'TALES' in trainingData or 'BLOGS' in trainingData:
        emo_maps['noemo']=6


    # Start main process    
    emoLabels=css.chooseEmoLabels(trainingData)
    final_paraphrases = translateAndscore(sourceLanguage, targetLanguages, numbForwtransl, numbBacktransl, trainingData, numTopParaphrases, decoding, goal, emoLabels)
    

    print('{} : {} : STATUS : End of the script.'.format(script, datetime.datetime.now()))
