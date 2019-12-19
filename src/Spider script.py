# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 14:33:09 2018

@author: dani

Takes trackmate data files and extracts x- and y-coordinates for heteromotility input
31 Aug 2018 version seems stable
"""

import pandas as pd
#from os import listdir
import csv
import itertools
import os
from random import randint
from random import seed as rseed
import datetime


input_dir = './'
outdir = input_dir

if not os.path.exists(outdir):
    os.makedirs(outdir)


file_array = [f for f in os.listdir(input_dir) if (f.startswith('Track stats'))]


for filename in file_array:
    outfile = outdir+filename.replace('Track stats','Prism_input')
    with open(outfile, 'w',newline='') as outfile:
        w = csv.writer(outfile)    
    
        df = pd.read_csv(input_dir + filename, usecols = ['Track ID', 'Position T', 'Position X ', 'Position Y'])
    
        # Remove centroids without an assigned track and sort for track id
    #    if df.TRACK_ID.dtype != 'int64':
    #        df = df[df.TRACK_ID != 'None'] 
    #        df.TRACK_ID = df.TRACK_ID.apply(pd.to_numeric)
    #    df = df.sort_values(['Track ID', 'Position T'])
        
        # create list of unique track numbers and initialize lists/dictionaries for data storage
        tracklist=df['Track ID'].unique().tolist()
        LenList = []
        xDict = {}
        yDict = {}
        
        # find data per track
        print (filename)
        for track in tracklist:
            print(track)
            trackdf = df.loc[df['Track ID'] == track].reset_index(drop=True)
            # create length lists per track
    #        t_points=len(track)
            x_0 = trackdf['Position X '][0]
            y_0 = trackdf['Position Y'][0]
#            
            # create x- and y- coordinates for each position in each track
            x = trackdf['Position X '] - x_0
            y = trackdf['Position Y'] - y_0
            
            # write x- and y-data to csv
    
            for i,pos in enumerate(x):
                row = [[x[i]],['None']*track,[y[i]]]
                row = list(itertools.chain.from_iterable(row))
                w.writerow(row)
                        
                