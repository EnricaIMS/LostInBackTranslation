'''
Reads XML files containing Sentence_id\tEmotion\tText, and converts them to different formats depending on the
goal of pipeline usage.

Returns filename_preprocessed.txt in folder data.
'''

import os, datetime, sys, configparser
import pandas as pd
from config import *

script='./scripts/preprocessing.py'

def convert_data(File, goal):

    myFile = pd.read_csv(File, sep="\t", header=None)
    colsINPUT = ["Sentence_id", "EmotionLabel", "Sentence"]
    myFile.columns = colsINPUT

    # extract emotion labels
    emotions = list(set(map(str.strip, myFile['EmotionLabel'].str.lower())))
    
    # insert column in file, depending on goal
    if goal == "Simple_Translation":
        target_emotion=['None']
    elif goal == "Recover_Emotion":
        print('ambe')    
    else:
        print('UAGLIO')
    new_column = target_emotion*len(myFile)
    myFile.insert(2, "Target_Emotion", new_column, allow_duplicates = False)
    
    myFile.to_csv(os.path.splitext(sys.argv[1])[0] + "_preprocessed.txt", index=False, sep="\t", header= None)
    print("{} : {} : STATUS : Preprocessed data have been saved in ./output/".format(script, datetime.datetime.now()))

if __name__ == "__main__":
    
    configParser = configparser.ConfigParser()
    configParser.read("config/config.ini")
    goal = configParser.get("Goal", "goal")

    convert_data(sys.argv[1], goal)
    
