#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Calls pipeline to produce backtranslations.
Checks the diff btw the emotion score of
backtranslations and input texts.
'''


import sys, os, datetime, configparser, re
import plot_results as plot
import matplotlib.pyplot as plt
import pandas as pd

script='./main.py'
path_to_Output='../output/'
path_to_Figure='../plots/'

def readConfig():
    config = configparser.ConfigParser()
    config.sections()
    config.read('../config/config.ini')
    parameters={}
    for section in config.sections():
        for options in config.options(section):
            parameters[section]=config.get(section, options)
            
    return(parameters)

    
def loadData(InputFile,trainData):
    '''
    Read scored_input and backtranslations files,
    produced by pipeline.py.
    '''
    original=pd.read_csv(InputFile, sep='\t',header=None)
                  
    scored=pd.read_csv(path_to_Output+'scored_input.txt',sep='\t',header=None)   
    paraphrases=pd.read_csv(path_to_Output+'scored_backtranslations.txt',sep='\t',header=None,quoting=3)

    # put data in the right format, depending on the corpus   
    if 'TEC' in trainData or 'TALES' in trainData or 'BLOGS' in trainData:
        colsSCOREDINPUT=['Sentence_id','ORIGINALemo', 'Sentence', 'surprise', 'joy', 'fear', 'anger', 'sadness', 'disgust']
    if 'ISEAR' in trainData:
        colsSCOREDINPUT=['Sentence_id','ORIGINALemo', 'Sentence', 'guilt','joy', 'fear', 'anger', 'sadness', 'disgust', 'shame']
    if 'TALES' in trainData or 'BLOGS' in trainData:
        colsSCOREDINPUT.append('noemo')
    colsINPUT=['Sentence_id','ORIGINALemo','TargetEmotions', 'Sentence']
    colsPAR=['Sentence_id','TARGETemo', 'Sentence', 'emoScore']
    original.columns=colsINPUT
    original=original.drop(columns=['TargetEmotions'])
    lowerEmo=original['ORIGINALemo'].str.lower()
    original['ORIGINALemo']=lowerEmo
    scored.columns=colsSCOREDINPUT
    lowerEmo=scored['ORIGINALemo'].str.lower() 
    scored['ORIGINALemo']=lowerEmo
    paraphrases.columns=colsPAR

    #split original sentences by emotion label
    emoNames=colsSCOREDINPUT[(colsSCOREDINPUT.index('Sentence')+1):]
    
    return(original, emoNames, scored, paraphrases)


def compareEmotionscores(original, emoNames, scored, paraphrases,top_n):
    '''
    Find deltas by emotion: check emotion change
    btw backtranslations and input texts.
    '''
    print('{} : {} : STATUS : Computing Delta scores by Emotion.'.format(script, datetime.datetime.now())) 
    allEmoDeltas={} #average Δs will be stored here
    for emo in emoNames:
        DELTAs={targetEmo:0 for targetEmo in emoNames} # Δs for the each emotion
   
        #input sentences labeled with that emotion
        originalSentences=original.loc[original['ORIGINALemo'] == emo]
        originalSentences.name=emo
        for i, row in originalSentences.iterrows():
            #Δ btw the emotions of a text and its backtranslations        
            deltaCurrentSentence= {targetEmo:0 for targetEmo in emoNames}            
            
            #take emotion scores of each sentence
            originalSentenceID=row['Sentence_id']
            originalScored=scored.loc[scored['Sentence_id'] == originalSentenceID]
            emoVector=originalScored.loc[:,emoNames]
            
            #take its backtranslations
            translations=paraphrases.loc[paraphrases['Sentence_id'] == originalSentenceID]

            for i, transl in translations.iterrows():
                target=transl['TARGETemo']
                score=transl['emoScore']
                #take score of the original sentence corresponding to target emo
                originalScore=emoVector[target].iloc[0]

                #####################################################
                #   COMPUTE DIFFERENCE BETWEEN NEW AND OLD SCORE    #
                #####################################################
                
                delta=round(score-originalScore,3)               
                deltaCurrentSentence[target]+=delta
                

            #average of Δs for the current sentence
            #(each input could have many translations, depending on n-best in config)
            deltaCurrentSentence = {k: v / top_n for k, v in deltaCurrentSentence.items()}            
            for key in deltaCurrentSentence.keys():
                DELTAs[key]+=round(deltaCurrentSentence[key],3)
            
        
        #Normalize the DELTA of emotion e by the sentences labeled as e
        try:
            DELTAs = {k: v / len(originalSentences) for k, v in DELTAs.items()}
        except ZeroDivisionError:
            print('{} : {} : STATUS : The input file contains no sentence labeled with this emotion: {}.'.format(script, datetime.datetime.now(),emo))
        
        allEmoDeltas[emo]=DELTAs
    
    allEmoDeltas=pd.DataFrame(allEmoDeltas).transpose() # input emotions on the rows

    return(allEmoDeltas)


def saveResults(scores,goal):
    outputFile=path_to_Output+'deltas.txt'
    scores.to_csv(outputFile, sep='\t') 
    plot.plotResults(goal)
    
def computeDeltas(InputFile):
    '''
    Combine above functions.
    '''
    parameters=readConfig()     
    trainingData=parameters['Data']
    goal=parameters['Goal']
    top_n=int(parameters['n_best'])
    
    mydata=loadData(InputFile,trainingData)  
    scores=compareEmotionscores(mydata[0], mydata[1], mydata[2], mydata[3],top_n)
    results=saveResults(scores,goal)
    print('{} : {} : STATUS : End of the script.'.format(script, datetime.datetime.now()))
 

if __name__ == "__main__":
    InputFile=sys.argv[1]       
    command='python pipeline.py -i '+InputFile
    os.system(command)

    evalutateBACKTRANSL=computeDeltas(InputFile)
