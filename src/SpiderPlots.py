# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 14:33:09 2018

@author: dani

- I want to add that the tracks used to make spider plots are exported into a new CSV which can then be used for heteromotility
- Add heteromotility implementation and analyses into single script as separate functions
  - Make these callable in the beginning of the script with if statements


"""

import pandas as pd
#from os import listdir
import csv
import itertools
import sys
import os
from random import randint
from random import seed as rseed
import datetime
import matplotlib.pyplot as plt


track_length = 60           # length interval of tracks to plot
pre_division_scrapping = 8  # time points before nuclear division to scrap
FOV = (-250,250)            # Field of view for spider plot
rseed(22)                   # random seed
lens = []
analysis_interv = 'random'  # must be 'first' OR 'last' OR 'random'
exp_name = 'VGM'            # only *.csv starting with these letters will be analyzed (can be empty string)



input_dir = './CF15/'
outdir = input_dir

if not os.path.exists(outdir):
    os.makedirs(outdir)


dirlist = [d for d in os.listdir(input_dir) if (os.path.isdir(input_dir+d) and d != outdir)]

for i,d in enumerate (dirlist):
    
    outfile_x = outdir+'HetMot_input_x_'+d+'.csv'
    outfile_y = outdir+'HetMot_input_y_'+d+'.csv'
#    xDict = {}
#    yDict = {}

    with open(outfile_x, 'w',newline='') as outfile_x, open(outfile_y, 'w',newline='') as outfile_y:
        wx,wy = csv.writer(outfile_x),csv.writer(outfile_y)
        
        # read input files
        file_array = [f for f in os.listdir(input_dir+d) if (f.startswith(exp_name) and f.endswith('.csv'))]        
            
        for filename in file_array:
        
            df = pd.read_csv(input_dir+d +'/'+ filename, usecols = ['Track ID', 'Position T', 'Position X ', 'Position Y', 'Frame']).rename(columns=lambda x: x.strip())
            df.columns = df.columns.str.replace(" ", "_")
            df.columns = map(str.upper, df.columns)
            
            # create list of unique track numbers and initialize lists/dictionaries for data storage
            tracklist=df.TRACK_ID.unique().tolist()
            LenList = []
            xDict = {}
            yDict = {}
            
            # find data per track
            for track in tracklist:
                trackdf = df.loc[df.TRACK_ID == track].reset_index(drop=True)
                
                #scrap split tracks (dividing cells) and X frames preceding division (X = pre_division_scrapping)
                if len(trackdf) != len(trackdf.FRAME.unique()):
                    framelist = list(trackdf.FRAME)
                    div_point = min([fr for fr in framelist if framelist.count(fr) > 1])-trackdf.FRAME[0]
                    analysis_end = max(0,div_point - pre_division_scrapping)
                    trackdf = trackdf[:analysis_end]
                
                if len(trackdf) != len(trackdf.FRAME.unique()):
                    print('duplicate timepoints still exist at timepoint %i, track %i, file: %s'%(div_point,track,filename))
                    sys.exit('fix duplicate track error')
                
                # add total length of tracks to list of lengths
                lens.append(len(trackdf))
                
                # create length lists per track
                if len(trackdf) > track_length:
                    t0 = randint(0, len(trackdf) - track_length)
                    
                    # create x- and y- coordinates for each position in each track.
                    if analysis_interv == 'random':
                        x,y = trackdf.POSITION_X[t0:t0+track_length],   trackdf.POSITION_Y[t0:t0+track_length]
                    elif analysis_interv == 'last':
                        x,y = trackdf.POSITION_X[-track_length:],       trackdf.POSITION_Y[-track_length:]
                    elif analysis_interv == 'first':
                        x,y = trackdf.POSITION_X[:track_length],        trackdf.POSITION_Y[:track_length]
                    
                    x = x - list(x)[0]
                    y = y - list(y)[0]
                    
                    # 0 total displacement in either x or y crashes heteromotility
                    if x.max()-x.min()>0 and y.max()-y.min() > 0:
                        
                        # plot track onto spider plot
                        plt.plot(x,y, linewidth=0.5,alpha=0.5)

                        # write files into heteromotility output files
                        wx.writerow([*round(x,3)])
                        wy.writerow([*round(y,3)])
#                        xDict[track] = x_track
#                        yDict[track] = y_track
            
#            # plot per file (= per movie)
#            plt.title(filename)
#            plt.xlim(-250,250)
#            plt.ylim(-250,250)
#            
#            plt.savefig(outdir+filename+'.png',  bbox_inches='tight',dpi=300)
#            plt.show()
#            plt.close()
    
    
        # plot per folder (= per condition)  
        plt.title(d)    
        plt.xlim(FOV)
        plt.ylim(FOV)
#        plt.legend()
    
        plt.savefig(outdir+d+'.png',  bbox_inches='tight',dpi=300)
        plt.show()
        plt.close()
                        
