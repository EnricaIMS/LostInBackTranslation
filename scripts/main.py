#!/usr/bin/env python3
# -*- coding: utf-8 -*-
########################################################################
#       THIS SCRIPT CHECKS THE DIFFERECE BETWEEN THE EMOTION SCORES    #
#           BETWEEN THE INPUT SENTENCES AND THE PARAPHRASES            #
#        This can be used also for other translation methods           #
#                                                                      #
#   How to:
#                python deltaEmo.py Translation input
#
#               (input: namefilecontainingsentences.txt)
#
#
########################################################################



import plot_results as plot
import analysis_module.st_dev_deltas as analysis_st_dev
import matplotlib.pyplot as plt
import configparser
import sys, os, datetime
import pandas as pd
import re

script='./deltaEmo.py'
path_to_Output='../outputs/'
path_to_Figure='../plots/'

def callEmoTransfer(ParaphraseModel,InputFile,emoClassifier):
    if ParaphraseModel=='Translation':
        print('OK')
        command='python emotion_transfer.py -c config.ini -i '+InputFile+' -o output.txt'
        os.system(command)
    else:
        print('{} : {} : STATUS : You have already the paraphrases, or you want to find another paraphrasing method'.format(script, datetime.datetime.now()))

def readConfig():
    config = configparser.ConfigParser()
    config.sections()
    config.read('config.ini')
    parameters={}
    for section in config.sections():
        for options in config.options(section):
            parameters[section]=config.get(section, options)
            
    return(parameters)

    


def loadData(InputFile,cluster,decoding):
    # load original sentences, their emotion scores and their paraphrases
    original=pd.read_csv(InputFile, sep='\t',header=None)
            
    global scoredFile
    try:
        scoredFile
    except NameError:
        scoredFile=re.search('[a-zA-Z]+.txt',sys.argv[2]).group()
           
    #Files produred by emotion_transfer.py       
    scored=pd.read_csv(path_to_Output+'fromfile/scored_'+scoredFile,sep='\t',header=None)   
    paraphrases=pd.read_csv(path_to_Output+'fromfile/paraphrases_'+decoding+'_'+cluster+'_'+scoredFile,sep='\t',header=None,quoting=3) #quoting controls whether quotes should be recognized. 3:QUOTE_NONNUMERIC
    # put data in the right format, depending on the corpus   
    if 'TEC' in scoredFile or 'TALES' in scoredFile or 'BLOGS' in scoredFile or 'DIALOGUES' in scoredFile:
        colsSCOREDINPUT=['Sentence_id','ORIGINALemo', 'Sentence', 'surprise', 'joy', 'fear', 'anger', 'sadness', 'disgust']
    if 'ISEAR' in scoredFile:
        colsSCOREDINPUT=['Sentence_id','ORIGINALemo', 'Sentence', 'guilt','joy', 'fear', 'anger', 'sadness', 'disgust', 'shame']
    if 'TALES' in scoredFile or 'BLOGS' in scoredFile or 'DIALOGUES' in scoredFile:
        colsSCOREDINPUT.append('noemo')

    
    colsINPUT=['Sentence_id','ORIGINALemo','TargetEmotions', 'Sentence']
    colsPAR=['Sentence_id','TARGETemo', 'Sentence', 'emoScore']
    
    original.columns=colsINPUT
    original=original.drop(columns=['TargetEmotions']) #this was used to store ALL target emotions, now useless
    
    lowerEmo=original['ORIGINALemo'].str.lower() #if emotions are uppercased, lower them
    original['ORIGINALemo']=lowerEmo
    
    scored.columns=colsSCOREDINPUT
    lowerEmo=scored['ORIGINALemo'].str.lower() 
    scored['ORIGINALemo']=lowerEmo
    
    paraphrases.columns=colsPAR
    print('{} : {} : STATUS : Loaded original sentences and their paraphrases.'.format(script, datetime.datetime.now()))
    
    
    print('{} : {} : STATUS : Dividing data by Original Emotion.'.format(script, datetime.datetime.now()))
    #split original sentences by emotion label
    emoNames=colsSCOREDINPUT[(colsSCOREDINPUT.index('Sentence')+1):]
    
    return(original, emoNames, scored, paraphrases)



'''FIND DELTA, RATIO AND AVERAGE BY EMOTION'''

def compareEmotionscores(original, emoNames, scored,paraphrases,cluster,decoding,top_n):  #both with difference and ratio
    
    #average deltas will be stored here
    allEmoDeltas={}
    allEmoRatios={}
    
    allValues_Input={} #store here the average of emotion scores
    allValues_Paraphrases={}
    
    f=open(path_to_Output+'fromfile/deltasBySentence'+decoding+'_'+cluster+scoredFile,'a')   
    deltaNames='\t'.join([re.sub('^','Δ_',emoname) for emoname in emoNames])    
    f.write('Sentence_id\tOriginal_emo\t'+deltaNames+'\n')
    
    
   
    print('{} : {} : STATUS : Computing Deltas, Ratios and Average scores by Emotion.'.format(script, datetime.datetime.now())) 
      
    for emo in emoNames:
        #store the Deltas and Ratios for the current emotion
        DELTAs={targetEmo:0 for targetEmo in emoNames} 
        RATIOs={targetEmo:0 for targetEmo in emoNames} 
        
        VALUEs_INPUT={targetEmo:0 for targetEmo in emoNames} 
        VALUEs_PARAPHR={targetEmo:0 for targetEmo in emoNames}
           
        #take all original sentences labeled with that emotion
        originalSentences=original.loc[original['ORIGINALemo'] == emo]
        originalSentences.name=emo
        
        #store scores here for boxplot
        df=pd.DataFrame(columns=emoNames)
    
        
        for i, row in originalSentences.iterrows():
            #compute DELTA and RATIO betwn the emotions of the sentence and its translations
            #and take the average emotion values of inputs and outputs
            
            deltaCurrentSentence= {targetEmo:0 for targetEmo in emoNames}
            ratioCurrentSentence= {targetEmo:0 for targetEmo in emoNames}
            
            val_CurrentSentence= {targetEmo:0 for targetEmo in emoNames}
            val_Paraphrases= {targetEmo:0 for targetEmo in emoNames}
            
            
            #retrieve the emotion scores of each sentence
            originalSentenceID=row['Sentence_id']
            originalScored=scored.loc[scored['Sentence_id'] == originalSentenceID]
            emoVector=originalScored.loc[:,emoNames]
            
            #retrieve its translations
            translations=paraphrases.loc[paraphrases['Sentence_id'] == originalSentenceID]
            if translations.empty:
                print('DataFrame is empty!')
                continue            
            
            f.write(str(originalSentenceID)+'\t'+emo+'\t')
            
           
            
            df2={}
            
            for i, transl in translations.iterrows():
                target=transl['TARGETemo']
                score=transl['emoScore']
                
                df2[target]=round(score,3)
                #this takes the score of the original sentence corresponding to target emo
                originalScore=emoVector[target].iloc[0]

                #####################################################
                #   COMPUTE DIFFERENCE BETWEEN NEW AND OLD SCORE    #
                #####################################################
                
                delta=round(score-originalScore,3)               
                deltaCurrentSentence[target]+=delta
                

                #####################################################
                #       COMPUTE RATIO BETWEEN NEW AND OLD SCORE     #
                #####################################################
                
                if originalScore!=0:
                    ratio=round(score/originalScore,3)
                    ratioCurrentSentence[target]+=ratio
                else:
                    continue
                                        
 
                #####################################################
                #      AVERAGE SCORES OF THE EMOTIONS OF INPUT      #
                #              AND PARAPHRASES                      #
                #####################################################
                       
                val_CurrentSentence[target]+=round(originalScore,3)   #value input
                val_Paraphrases[target]+=round(score,3) #value its paraphrases
                    
            #put emo values in dictionary
            df = df.append(df2, ignore_index=True)
            
            #average of delta for the current sentence
            #because each original could have many translations
            deltaCurrentSentence = {k: v / top_n for k, v in deltaCurrentSentence.items()}
            
            DeltasTostring='\t'.join([str(x) for x in deltaCurrentSentence.values()])
            f.write(DeltasTostring+'\n')

            
            for key in deltaCurrentSentence.keys():
                DELTAs[key]+=round(deltaCurrentSentence[key],3)
            
            
            
            #same for ratio
            ratioCurrentSentence = {k: v / top_n for k, v in ratioCurrentSentence.items()}
            for key in ratioCurrentSentence.keys():
                RATIOs[key]+=round(ratioCurrentSentence[key],3)
                
                
            #for abs values of current input, no need to divide by numb of top_n      
            val_Paraphrases={k: v / top_n for k, v in val_Paraphrases.items()}
            for key in val_Paraphrases.keys():
                VALUEs_PARAPHR[key]+=round(val_Paraphrases[key],3)
            for key in val_CurrentSentence.keys():
                VALUEs_INPUT[key]+=round(val_CurrentSentence[key],3) 
            
        #make boxplot of emo values
        bpName=re.sub('.txt','',scoredFile)
        myFig = plt.figure()
        bp = df.boxplot()
        myFig.savefig(path_to_Figure+emo+'_'+decoding+'_'+cluster+bpName+".pdf", format="pdf")
        
        
        #Divide the DELTA of emotion by the number of sentences labeled with EMOTION emo
        try:
            DELTAs = {k: v / len(originalSentences) for k, v in DELTAs.items()}
        except ZeroDivisionError:
            print('{} : {} : STATUS : The input file contains no sentence labeled with this emotion: {}.'.format(script, datetime.datetime.now(),emo))
        
        allEmoDeltas[emo]=DELTAs
        
        #Same for ratios
        try:
            RATIOs = {k: v / len(originalSentences) for k, v in RATIOs.items()}
        except ZeroDivisionError:
            print('{} : {} : STATUS : The input file contains no sentence labeled with this emotion: {}.'.format(script, datetime.datetime.now(),emo))
        allEmoRatios[emo]=RATIOs
        
        #Same for abs values
        try:
            #len(originalSentences) and not translations because we have one translation per emotion
            #for a total of len(originalSentences) per emotion
            VALUEs_PARAPHR = {k: v / len(originalSentences) for k, v in VALUEs_PARAPHR.items()}
        except ZeroDivisionError:
            print('{} : {} : STATUS : The input file contains no sentence labeled with this emotion: {}.'.format(script, datetime.datetime.now(),emo))
        allValues_Paraphrases[emo]=VALUEs_PARAPHR
        
        VALUEs_INPUT= {k: v / len(originalSentences) for k, v in VALUEs_INPUT.items()}
        allValues_Input[emo]=VALUEs_INPUT
        
    
    
    #Original emotions are on the rows
    allEmoDeltas=pd.DataFrame(allEmoDeltas).transpose()
    allEmoRatios=pd.DataFrame(allEmoRatios).transpose()

    allValues_Input=pd.DataFrame(allValues_Input).transpose()
    allValues_Paraphrases=pd.DataFrame(allValues_Paraphrases).transpose()
    
    f.close()
    
    return(allEmoDeltas,allEmoRatios,allValues_Input,allValues_Paraphrases)


        
''' SAVE RESULTS'''
def saveResults(scores,cluster,decoding,goal):
    print('{} : {} : STATUS : Saving Deltas results.'.format(script, datetime.datetime.now()))
    #Plot
    outputFile=path_to_Output+'fromfile/deltas-'+decoding+'_'+cluster+'_'+scoredFile
    scores[0].to_csv(outputFile, sep='\t') 
    plot.plotResults(decoding,cluster,scoredFile,'deltas',goal)
    
    print('{} : {} : STATUS : Saving Ratios results.'.format(script, datetime.datetime.now()))
    outputFile=path_to_Output+'fromfile/ratios-'+decoding+'_'+cluster+'_'+scoredFile
    scores[1].to_csv(outputFile, sep='\t')     
    plot.plotResults(decoding,cluster,scoredFile,'ratios',goal)
    
    print('{} : {} : STATUS : Plotting deltas by emotion.'.format(script, datetime.datetime.now()))
    plot.plot_deltas_by_emotion(decoding,cluster,scoredFile)
    
    #Save stdev of deltas
    print('{} : {} : STATUS : Finding standard deviation of deltas.'.format(script, datetime.datetime.now()))
    analysis_st_dev.st_dev_by_emotion(decoding,cluster,scoredFile)
    plot.plotResults(decoding,cluster,scoredFile,'standard_deviation_deltas',goal)

''' COMBINE FUNCTIONS'''
def computeDeltas(ParaphraseModel,InputFile):
    
    parameters=readConfig()     
    emoClassifier=parameters['emotionClassifier']          
    cluster=parameters['Cluster']
    decoding=parameters['Decoding']
    goal=parameters['Goal']
    top_n=int(parameters['n_best']) #this is the numb of paraphrases per sentence
    
    emotransfer=callEmoTransfer(ParaphraseModel,InputFile,emoClassifier)
    mydata=loadData(InputFile,cluster,decoding)  
    scores=compareEmotionscores(mydata[0], mydata[1], mydata[2], mydata[3],cluster,decoding,top_n)
    results=saveResults(scores,cluster,decoding,goal)
    print('{} : {} : STATUS : End of the script.'.format(script, datetime.datetime.now()))
 

''' MAIN METHOD'''
if __name__ == "__main__":
    ParaphraseModel=sys.argv[1]
    InputFile=sys.argv[2]       

    evalutateBACKTRANSL=computeDeltas(ParaphraseModel,InputFile)
