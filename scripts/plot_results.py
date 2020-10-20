#!/usr/bin/env python3
'''
Visualizes delta scores as heatmaps.
'''
import datetime
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
#from matplotlib.ticker import LogFormatter
from matplotlib.colors import LogNorm, SymLogNorm

script='./plot_results.py'
path_to_Output='../output/'
path_to_Figure='../fig/'

def plotResults(goal):
    
    print('{} : {} : STATUS : Plotting results.'.format(script, datetime.datetime.now()))

    nameofFile=path_to_Output+'deltas.txt'  
    results=pd.read_csv(nameofFile,sep='\t') 
    # SET CORRECT INDEX AND COLUMNS FOR DATAFRAMES
    # input emotions will be on the rows
    with open(nameofFile,'r') as myf:
        cols=myf.readlines()[0]
        cols=cols.split()
        cols.insert(0,'original_emotions')
    results.columns=cols
    results.set_index('original_emotions')
    rows=list(results['original_emotions'])
    results=results.drop(columns=['original_emotions'])

    df = pd.DataFrame(results.values, index=rows, columns=cols[1:])
    df=df.round(3)
    
    df = df.sort_index(axis=1)
    df = df.sort_index(axis=0)

    df.style.background_gradient(cmap='Blues')
    plt.figure()

    if goal=='Simple_Overgeneration':
        log_norm=LogNorm(vmin=0, vmax=.5)
    if goal=='Restore_Overgeneration':
        log_norm= SymLogNorm(linthresh=0.001, vmin=-.1,vmax=1.)
    else:
        log_norm= SymLogNorm(linthresh=0.001, vmin=-.04, vmax=.35)
    sns_plot = sns.heatmap(df, annot=True, norm=log_norm, cmap="YlGnBu")
    
    figureName=path_to_Figure+'deltas.pdf'
    plt.savefig(figureName,bbox_inches="tight")

    print('{} : {} : STATUS : Saved plots for {}.'.format(script, datetime.datetime.now(), nameofFile))
