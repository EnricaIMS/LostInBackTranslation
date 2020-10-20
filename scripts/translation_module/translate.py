#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Translate files. Decoding method and 
numb. of forward and backward hypotheses 
are in configs/config.ini
'''

import torch, re, string
import sys, datetime, random
from io import StringIO
from nltk.tokenize import sent_tokenize
from translation_module.interactiveNMT import iTranslate #this serves to load the models from a specific path


script='./translation_module/translate.py'
   
class Translation:
    
    def loadTranslationModel(self,sourceLanguage:str,targetLanguages:list,decoding:str,numForward:int,numBackward:int):
        path='../../../backtranslation/scripts/'
        global forwardpairs
        global backwardpairs
        forwardpairs={}
        backwardpairs={}
        #if we're just taking the 1-best translation
        if numForward==1 and numBackward==1:
            if decoding=='Beamsearch':
                configuration='--beam 5 --nbest 1'
            else:
                configuration=' --sampling --sampling-topk 10 --beam 1 --nbest 1'

        else:
            configuration=' --beam 100 --nbest 100'
            #Sampling
            if decoding=='Sampling':
                configuration=' --sampling --sampling-topk 10'+configuration

        #load the models
        for t in targetLanguages:
            forward='--path '+path+'translation_module/translation_models/wmt19.'+sourceLanguage+'-'+t+'.joined-dict.ensemble/model4.pt '+path+'translation_module/translation_models/wmt19.'+sourceLanguage+'-'+t+'.joined-dict.ensemble/ --source-lang '+sourceLanguage+' --target-lang '+t+' --remove-bpe --bpe fastbpe --bpe-codes '+path+'translation_module/translation_models/wmt19.'+sourceLanguage+'-'+t+'.joined-dict.ensemble/bpecodes '+configuration+' --tokenizer moses'
            backward='--path '+path+'translation_module/translation_models/wmt19.'+t+'-'+sourceLanguage+'.joined-dict.ensemble/model4.pt '+path+'translation_module/translation_models/wmt19.'+t+'-'+sourceLanguage+'.joined-dict.ensemble/ --source-lang '+t+' --target-lang '+sourceLanguage+' --remove-bpe --bpe fastbpe --bpe-codes '+path+'translation_module/translation_models/wmt19.'+t+'-'+sourceLanguage+'.joined-dict.ensemble/bpecodes '+configuration+' --tokenizer moses'
            forwardpairs[sourceLanguage+'2'+t]=iTranslate(forward)
            print('{} : {} : STATUS : Loaded model for forward translation.'.format(script, datetime.datetime.now()))

            backwardpairs[t+'2'+sourceLanguage]=iTranslate(backward)
            print('{} : {} : STATUS : Loaded model for backward translation.'.format(script, datetime.datetime.now()))        

            
            
     
    def forwardbackTranslation(self,inputText:str,numForward:int,numBackward:int,sourceLanguage:str,targetLanguages:list):
        translations=list()
        
        # Translate current sentence with the models that have already been uploaded
        for f in forwardpairs.keys(): #consider all the language pairs of forward translation (e.g. en->de, en->ru)
            
        
            languages=f.split('2')
            target=languages[1]
            source=languages[0]
            
            back=target+'2'+source # direction of backtranslation
            
            inputText=sent_tokenize(inputText)

            print('{} : {} : STATUS : The Translation Model will now translate the input text from {} to {}.'.format(script, datetime.datetime.now(),source,target))
            
            #the backtranslations for one input
            translation=list()
            
            # translate sentence by sentence
            for sentence in inputText:
                sentence=re.sub('"','',sentence)
                sentence=sentence.replace('\\','').strip()
                current_sentence=list()
                
                '''
                Go forward: Translate sentence by sensence.
                Try to take numForward uniques translations. 
                '''
                forwardTranslation=forwardpairs[f].translate_batch([sentence])
        
                uniques_forward={} # (original sentence:stripped sentence) later used for backtranslation
                
                for line in forwardTranslation:
                    # If two sentences are the same (regardless of punctuation and upper/lowercase), consider only 1
                    stripped=line.translate(str.maketrans('', '', string.punctuation)) 
                    u=stripped.lower().strip()
                    if u not in uniques_forward.values():
                        uniques_forward[line]=u
                    
                forwardSentences=list(uniques_forward.keys())[:numForward]
                #but if this is not possible, take some (# of missing sentences to reach the numForward) from cleanForward
                if len(forwardSentences)<numForward:
                    missings=numForward-len(forwardSentences)
                    while len(forwardSentences)<numForward:
                        if missings>len(forwardSentences): #there are more missing sentences than sentences in forwardtranslation
                            randSentences=[random.choice(forwardSentences)]
                        else: 
                            missings=numForward-len(forwardSentences)
                            randSentences=random.sample(forwardSentences,missings)
                        for r in randSentences:
                            forwardSentences.append(r)
                '''
                Go backward.
                '''
                back_uniques={}
                for sentence in forwardSentences:
         
                    backTranslation=backwardpairs[back].translate_batch([sentence])
                     
                    #Take only some of them (uniques numBackward)
                    uniques_backward={}
                    for line in backTranslation:
                        if len(uniques_backward.keys())<numBackward:
                            stripped=line.translate(str.maketrans('', '', string.punctuation)) # strip from punctuation
                            s=stripped.lower().strip()
                            if s not in back_uniques.values() and len(uniques_backward)<numBackward:
                                uniques_backward[line]=(s)
                                back_uniques[line]=(s) 
                             
                            else:
                                continue
                    if len(uniques_backward)==0: #the backtranslations for the current sentence have already been found for the others
                        backSentences=backTranslation[:numBackward] #so just take some backtranslations, even if not unique    
                    else:
                        backSentences=list(uniques_backward.keys())[:numBackward]
                    print(backSentences)
    
                    if len(backSentences)<numBackward:
                        while len(backSentences)<numBackward:
                            missings=numBackward-len(backSentences)
                            if missings>len(backSentences): #there are more missing sentences than sentences in backtranslation
                                randSentences=[random.choice(backSentences)]
                            else:
                                randSentences=random.sample(backSentences,missings)
                            for r in randSentences: #missing sentences can be picked from backSentences
                                backSentences.append(r)
            
                    current_sentence.append(backSentences)
                
                current_sentence=[item for sublist in current_sentence for item in sublist]
                translation.append(current_sentence)
            
            if len(inputText)!=1:
                #merge the sentence by sentence translations
                merge_translation=list(zip(*translation))
            
                translation=[' '.join(list(t)) for t in merge_translation]
            else:
                translation=translation[0]
            

            #check if we have correct number of backtranslations
            target_totals=len(forwardSentences)*len(backSentences)
            if len(translation)<target_totals:
                with open('../outputs/fromfile/Sentences_with_missing_translations.txt','w') as missing_tr:
                    #if not, add the sentence itself as translation and save that
                    missings=target_totals-len(translation)
                    missing_tr.write(inputText+' : added as a backtranslation'+str(len(missing))+'times.\n')
                    for r in range(missing):
                        translation.append(inputText)
                        
                

            
            for y in translation:
                translations.append([y,target])     
            print('{} : {} : STATUS : Produced {} backward translations for the current sentence'.format(script, datetime.datetime.now(), len(translations)))
                              
                        
        print('{} : {} : STATUS : Done with backtranslation.'.format(script, datetime.datetime.now()))
        
        return translations
