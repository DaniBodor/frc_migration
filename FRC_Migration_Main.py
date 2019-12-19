# -*- coding: utf-8 -*-
"""

@author: dani
stable version as per 13 March, 2019

# Once per computer, Before the first run, install heteromotility by copying the following line into the console (and hit enter).
! pip install heteromotility


# Minor error is the vertical position of the text in the boxplots, which is currently only dependent on the max value,
# rather than the range. this results in incorrect positions if min value is not (approx) 0 or if it goes into negative
   I will probably not fix this, unless for publication purposes, in which case contact me (Dani) to do so


# Non-essential, but for nicer heteromotility output files that don't have unnecessary white lines
# change the file hmio.py line 90 (or thereabouts in future versions)
# from
    with open(output_file_path, 'w') as out_file:
# to
    with open(output_file_path, 'w',newline='') as out_file:
# make sure the correct number of white spaces at start of line is maintained
    
"""

#%% Set parameters

# set which script modules to run (set to True/1 or False/0)
MOD_spiderplots = 1         # create spider plots
MOD_run_HM = 0              # runs heteromotility (on files created above)
MOD_DimReduct = 0           # PCA and tSNE analysis of data produced by heteromotility 
MOD_ComparisonPlots = 0     # generate plots for individual parameters of heteromotility 

# input names that need to be changed or checked for each experiment
#base_dir = 'CF19'           # working directory; this script should be saved in the same folder as base_dir    #obsolete!
exp_folder = 'CF15'   # folder with 1 subfolder per condition with the csv files to be analyzed
exp_name = ''               # only *.csv starting with these letters will be analyzed (can be empty string to analyze everything)

# parameters that need can be changed to reflect settings/choices of analysis
track_length = 60           # length interval of tracks to plot/analyze 
pre_division_scrapping = 8  # time points before nuclear division to scrap
analysis_interv = 'random'  # must be 'first' OR 'last' OR 'random'
move_thresh = 1             # for heteromotility: highest speed to check as a threshold for movement in time_moving and avg_moving_speed parameters (HM default is 10); unsure whether this is in um/frame or px/frame
speed_thresh = 0            # minimum threshold of movement speed to be included in dimension reduction (can be 0); unsure whether this is in um/frame or px/frame
rseed = 22                  # random seed (to produce reproducible quasi-ramdom numbers)

# parameters relating to data visualizations
plot_size = 250             # size of spider plot
spider_type = 'condition'   # must be 'condition' or 'movie' reflecting what each speiderplot represents
p_dec = 4                   # number of decimals shown in p value of ANOVA and Kruskal-Wallis    

# Choose features from heteromotility to be used for PCA and t-SNE
# Use Ctrl+1 to include/exclude entire lines, or use #s to remove everything behind
# make sure each line ends with a comma, and that there's a closing square bracket at the very end
DimReductParas = [#'Well/XY', 'cell_id',  
       'total_distance', 'net_distance', 'linearity', 'spearmanrsq', 'progressivity',
       'max_speed', 'min_speed', 'avg_speed',
       'MSD_slope', 'hurst_RS', 'nongauss', 'disp_var', 'disp_skew',
       'rw_linearity', 'rw_netdist',
       'rw_kurtosis01', 'rw_kurtosis02', 'rw_kurtosis03', 'rw_kurtosis04', 'rw_kurtosis05',
       'rw_kurtosis06', 'rw_kurtosis07', 'rw_kurtosis08', 'rw_kurtosis09', 'rw_kurtosis10',
       'avg_moving_speed01', 'avg_moving_speed02', 'avg_moving_speed03', 
       'avg_moving_speed04', 'avg_moving_speed05',
       'avg_moving_speed06', 'avg_moving_speed07', 'avg_moving_speed08', 'avg_moving_speed09', 'avg_moving_speed10',
       'time_moving01', 'time_moving02', 'time_moving03', 
       'time_moving04', 'time_moving05',
       'time_moving06', 'time_moving07', 'time_moving08', 'time_moving09', 'time_moving10',
       'autocorr_1', 'autocorr_2', 'autocorr_3', 'autocorr_4', 'autocorr_5',
       'autocorr_6', 'autocorr_7', 'autocorr_8', 'autocorr_9', 'autocorr_10',
#       'p_rturn_9_5', 'p_rturn_9_6', 'p_rturn_10_5', 'p_rturn_10_6', 'p_rturn_11_5', 'p_rturn_11_6',
#       'mean_theta_9_5', 'min_theta_9_5', 'max_theta_9_5', 'mean_theta_9_6', 'min_theta_9_6', 'max_theta_9_6',
#       'mean_theta_10_5', 'min_theta_10_5', 'max_theta_10_5', 'mean_theta_10_6', 'min_theta_10_6', 'max_theta_10_6',
#       'mean_theta_11_5', 'min_theta_11_5', 'max_theta_11_5', 'mean_theta_11_6', 'min_theta_11_6', 'max_theta_11_6',
       ]

# output folder names
spider_outdir = 'X_SpiderPlots'   # folder that contains spiderplots
HM_indir = 'X_HM_input'           # folder that contains (x,y)-coordinates for heteromotility
HM_outdir = 'X_HM_output'         # folder that contains output data from heteromotility
Quant_analys = 'X_QuantAnalysis'  # folder that contains graphs for PCA, tSNE, and individual parameters

# parameters for tracking profress of script (don't really need to change)
show_spiderplots = False    # turn on to visualize spiderplots in console (mainly for debugging purposes)
print_HM_command = False    # turn on to see the name of command line callout for heteromotility for each file in console (mainly for debugging purposes)
print_HM_process = True     # turn on to see the process of heteromotility in console

#%% default imports and parameters
import pandas as pd
#import itertools
import sys
import os
from datetime import datetime
import matplotlib.pyplot as plt
from random import seed as rseed

starttime = datetime.now()
rseed(rseed)
outdir_list = []
lens = []
counter = 0

#base_dir = './%s/'%base_dir
base_dir = './'         #
input_dir = base_dir + 'data/raw/' + exp_folder + '/'
base_outdir = base_dir + 'data/processed/ ' + f'tracklength{track_length}_movethresh{move_thresh}/'


spider_outdir = base_outdir + spider_outdir + '/'
HM_indir = base_outdir+HM_indir+ '/'
HM_outdir = base_outdir+HM_outdir+ '/'
Quant_analys = base_outdir+Quant_analys+ '/'

MOD_HM_input = MOD_HM_input = 1           # generates files that can be read by heteromotility

if MOD_spiderplots:
    outdir_list.append(spider_outdir)
if MOD_HM_input:
    outdir_list.append(HM_indir)
if MOD_run_HM:
    outdir_list.append(HM_outdir)
if MOD_DimReduct or MOD_ComparisonPlots:
    outdir_list.append(Quant_analys)

for outdir in outdir_list:
    if not os.path.exists(outdir):
        os.makedirs(outdir)

input_para_set = [
        base_dir , exp_folder , exp_name , 
        track_length , pre_division_scrapping , analysis_interv , move_thresh , speed_thresh , rseed ,
        plot_size , spider_type , p_dec,
        DimReductParas,
        ]


log = f'''
parameters for run on {starttime}

input data:
    base_dir = {base_dir}
    exp_folder = {exp_folder}
    exp_name = {exp_name}

analysis settings: 
    track_length = {track_length}
    pre_division_scrapping = {pre_division_scrapping}
    analysis_interv = {analysis_interv}
    move_thresh = {move_thresh}
    speed_thresh = {speed_thresh}
    rseed = {rseed}

visualization:
    plot_size = {plot_size}
    spider_type = {spider_type}
    p_dec = {p_dec}

heteromotility parameters to include in dimenstion reduction:
    {[p for p in DimReductParas]}
'''
with open(base_outdir+"log.txt", "w") as text_file:
    print(log, file=text_file)


#%% custom defined functions
def scrap_split_tracks(trackdf,pre_division_scrapping):
    '''
    scraps split tracks (usually dividing cells) 
    as well as X frames split track (cells preparing to divide)
    EXITS CODE if duplicate tracks are still found (fix these manually and rerun code)
    '''
    if len(trackdf) != len(trackdf.FRAME.unique()):
        framelist = list(trackdf.FRAME)
        div_point = min([fr for fr in framelist if framelist.count(fr) > 1])-trackdf.FRAME[0]
        analysis_end = max(0,div_point - pre_division_scrapping)
        trackdf = trackdf[:analysis_end]
    
#    if len(trackdf) != len(trackdf.FRAME.unique()):
    if len(trackdf) != len(trackdf.FRAME.unique()):
        print('duplicate timepoints still exist at timepoint %i, track %i, file: %s'%(div_point,track,filename))
        raise IndexError('duplicate timepoints exist')  #this needs to be fixed, probably by a custom error
#    except:
        # yadda yadda fix this dskbfsdk
    
    return trackdf


#%% spiderplots and (x,y)-coordinate extraction
if MOD_spiderplots or MOD_HM_input:
    SPstart = datetime.now()
    print ('starting on spiderplot creation and/or (x,y)-data extraction')
    import csv
    from random import randint
    FOV = (-plot_size,plot_size)
    
    # loop through subdirectories in input dir (excluding output directories)
    dirlist = [d for d in os.listdir(input_dir) if (os.path.isdir(input_dir+d) and d not in outdir_list)]
    for i,d in enumerate (dirlist):
        outfile_x = HM_indir+d+'_x.csv'
        outfile_y = HM_indir+d+'_y.csv'
        
        # read input files
        file_array = [f for f in os.listdir(input_dir+d) if (f.startswith(exp_name) and f.endswith('.csv'))]     
        xDict = {}
        yDict = {}
            
        for filename in file_array:
        
            spider_df = pd.read_csv(input_dir+d +'/'+ filename, usecols = ['Track ID', 'Position T', 'Position X ', 'Position Y', 'Frame']).rename(columns=lambda x: x.strip())
            spider_df.columns = spider_df.columns.str.replace(" ", "_")
            spider_df.columns = map(str.upper, spider_df.columns)
            
            # create list of unique track numbers and initialize lists/dictionaries for data storage
            tracklist=spider_df.TRACK_ID.unique().tolist()
            LenList = []
            
            # find data per track
            for track in tracklist:
                trackdf = spider_df.loc[spider_df.TRACK_ID == track].reset_index(drop=True)
                
                #scrap split tracks (dividing cells) and X frames preceding division (X = pre_division_scrapping)
                trackdf = scrap_split_tracks(trackdf,pre_division_scrapping)          
                
                # add total length of tracks to list of lengths
                lens.append(len(trackdf))
                
                # create length lists per track
                if len(trackdf) > track_length:
                    t0 = randint(0, len(trackdf) - track_length)
                    
                    # create x- and y- coordinates for each position in each track.
                    if analysis_interv == 'random':
                        x_coord,y_coord = trackdf.POSITION_X[t0:t0+track_length],   trackdf.POSITION_Y[t0:t0+track_length]
                    elif analysis_interv == 'last':
                        x_coord,y_coord = trackdf.POSITION_X[-track_length:],       trackdf.POSITION_Y[-track_length:]
                    elif analysis_interv == 'first':
                        x_coord,y_coord = trackdf.POSITION_X[:track_length],        trackdf.POSITION_Y[:track_length]
                    
                    x_coord,y_coord = x_coord - list(x_coord)[0], y_coord - list(y_coord)[0]
                    
                    # 0 total displacement in either x or y crashes heteromotility
                    if x_coord.max()-x_coord.min()>0 and y_coord.max()-y_coord.min() > 0:
                        counter +=1
#                            print (counter, d, filename, track)

                        
                        # plot track onto spider plot
                        if MOD_spiderplots:
                            plt.plot(x_coord,y_coord, linewidth=0.5,alpha=0.5)

                        # write files into heteromotility output files
                        if MOD_HM_input:
#                                wx.writerow([*round(x_coord,3)])
#                                wy.writerow([*round(y_coord,3)])
                            xDict[counter] = x_coord
                            yDict[counter] = y_coord
            
            # plot per file (= per movie)
            if spider_type == 'movie' or spider_type == 'file':
                plt.title(filename)
                plt.xlim(FOV)
                plt.ylim(FOV)
                
                plt.savefig(spider_outdir+'spider_'+filename+'.png',  bbox_inches='tight',dpi=300)
                if show_spiderplots:
                    plt.show()
                plt.close()
    
    
        # plot per folder (= per condition)  
        if spider_type == 'condition' or spider_type == 'folder':
            plt.title(d)
            plt.xlim(FOV)
            plt.ylim(FOV)
        
            plt.savefig(spider_outdir+'spider_'+d+'.png',  bbox_inches='tight',dpi=300)
            if show_spiderplots:
                plt.show()
            plt.close()
        
        if MOD_HM_input:
            with open(outfile_x, 'w',newline='') as outfile_x, open(outfile_y, 'w',newline='') as outfile_y:
                wx,wy = csv.writer(outfile_x),csv.writer(outfile_y)
                for xkey,xvalue in xDict.items():
                    wx.writerow([*round(xvalue,3)])
                for ykey,yvalue in yDict.items():
                    wy.writerow([*round(yvalue,3)])
    
    SPfinish = datetime.now()
    SPtime = int(round((SPfinish - SPstart).total_seconds(),0))
    print ('spiderplots and/or (x,y)-data extraction done after %i seconds'%SPtime)
    print ('')
    
#%% run heteromotility                        
if MOD_run_HM:
    HMstart = datetime.now()
    print ('starting to run heteromotility')
    import subprocess as sp
    
    file_array = os.listdir(HM_indir)
    outdir_files = os.listdir(HM_outdir)
    
    skip_count = 0
    
    for name in file_array:
        if 1==0:        # replace by line below if I want to skip files already processed
#        if 'HMout_'+name[:-6]+'.csv' in outdir_files:
    #        print ('already processed: '+name)
            q=0
        elif name.endswith('_x.csv'):
            x_file = os.path.abspath(HM_indir)+name
            y_file = x_file.replace('_x.csv','_y.csv')
            
            if os.path.exists(y_file):
                base_name = name[:-6]
                
                # will print the status and commandline
                if print_HM_command:
                    print  ( 'heteromotility.exe',os.path.abspath(HM_outdir),'--tracksX',x_file,'--tracksY',y_file,'--output_suffix',base_name,'--move_thresh',str(move_thresh))
                    
                # run heteromotility as if in BASH
                sys.exit('fsdjhfjsd')
                print ('running heteromotility on ' + base_name)
                sp.call(['heteromotility.exe',os.path.abspath(HM_outdir),'--tracksX',x_file,'--tracksY',y_file,'--output_suffix',base_name,'--move_thresh',str(move_thresh)])
                    
            elif print_HM_process:
                print (y_file + "doesn't exist")
                print('--- skipped')
                skip_count+=1

    HMfinish = datetime.now()
    HMtime = int(round((HMfinish - HMstart).total_seconds(),0))
    print ('heteromotility processing finished after %i seconds'%HMtime)
    print ('')

#%% run dimension reduction
if MOD_DimReduct:
    DRstart = datetime.now()
    print ('starting on dimension reductions')
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
#    from scipy.stats import  mannwhitneyu, mstats
    import matplotlib
    import csv
     
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    
    colors=['xkcd:azure','xkcd:darkblue','xkcd:cyan',
            'xkcd:sienna','brown',
    #        'xkcd:orange','xkcd:red'
            ]
    #colors=[0,1,2,3,4]
    
    
    HM_files = os.listdir(HM_outdir)
    
    # Read in the csv in pandas, adds a label column = filename, drops all cells slower than speed_thresh
    samples = []
    for filename in HM_files:
        HM_df = pd.read_csv(HM_outdir + filename , usecols = DimReductParas)
        HM_df['label'] = filename
        samples.append(HM_df[HM_df.avg_speed>speed_thresh])
    
    with open(HM_outdir + filename) as csvFile:
        reader = csv.reader(csvFile)
        DimReductParas = next(reader)[1:]

    # All samples will be put together in df and label column is dropped from df
    DRdf = pd.concat(samples, ignore_index = True)
    label_df = DRdf.label
    DRdf = DRdf.drop(['label'], axis = 1)
    
    # Data is normalized/scaled to ensure equal contribution from all features
#    normalized_df = (DRdf - DRdf.min()) / (DRdf.max() - DRdf.min())    #sascha's method, depends heavily on outliers!
    normalized_df = (DRdf - DRdf.mean()) / DRdf.std()
    
    # Create a PCA object from sklearn.decomposition
    pca = PCA()
    # Perform PCA on the normalized data and return transformed principal components
    transformed = pca.fit_transform(normalized_df.values)
    components = pca.components_
    normed_comp = abs(components)/np.sum(abs(components),axis = 0)
    
    # Calculate variance contribution of each principal component (currently unused)
    expl_var_ratio = pca.explained_variance_ratio_
    
    # Create a scatter plot of the first two principal components
    w, h = plt.figaspect(1.)
    pca_fig, pca_ax =plt.subplots(figsize=(w,h))
    for x,i in enumerate(HM_files):
        pca_ax.scatter(transformed[:,0][label_df == i], transformed[:,1][label_df == i], 
                    label = i[6:-4], alpha=0.5, s=5, )
        #, c=colors[x])
    
    # Format PCA graph
    pca_ax.legend(#loc='upper center', bbox_to_anchor=(0.5, 1.05),
#          ncol=math.ceil(len(HM_files)/2), 
          fancybox=True, shadow=False, prop={'size': 6}, 
          framealpha=0.75)
    pca_ax.spines['right'].set_visible(False)
    pca_ax.spines['top'].set_visible(False)
    pca_ax.set_xlabel('PC1 (variance ' + str(int(expl_var_ratio[0]*100))+ ' %)')
    pca_ax.set_ylabel('PC2 (variance ' + str(int(expl_var_ratio[1]*100))+ ' %)')
    
    # Save PCA plot
    pca_fig.savefig(Quant_analys + '0_PCA.png',  bbox_inches='tight',dpi=1200)
#        pca_fig.savefig(base + '__' + thisdir + '_pca_.pdf',  bbox_inches='tight')
    plt.close()
    DR_halfway = datetime.now()
    DRtime = int(round((DR_halfway - DRstart).total_seconds(),0))
    print ('PCA done after %i seconds'%DRtime)

    
    # Create t-SNE plot without axis
    tsne = TSNE(n_components = 2, init = 'pca', random_state= 0 )
    tsne_points = tsne.fit_transform(normalized_df.values)
    fig, ax = plt.subplots(figsize=(w,h))
    ax.axis('off')

    for x,i in enumerate(HM_files):
        ax.scatter(tsne_points[:,0][label_df == i], tsne_points[:,1][label_df == i], 
               label = i[6:-4], alpha=0.5, s=5,)
#                   c=colors[x])

    # Format tSNE graph    
    ax.legend(#loc='upper left', #bbox_to_anchor=(0.5, 1.05),
          ncol=1, fancybox=True, shadow=False, prop={'size': 6},
          framealpha=0.75)
    
    # Save t-SNE plot in directory
    fig.savefig(Quant_analys + '0_tSNE.png',  bbox_inches='tight',dpi=1200)
#        fig.savefig(base + '__' + thisdir + '_tsne_.pdf',  bbox_inches='tight')
    
    plt.close()
    
    DRfinish = datetime.now()
    DRtime = int(round((DR_halfway - DRstart).total_seconds(),0))
    print ('tSNE done after %i seconds'%DRtime)
    print ('')

#%% create individual plots for each parameter in heteromotility
if MOD_ComparisonPlots:
    CPstart = datetime.now()
    print ('starting on boxplots')
    import seaborn as sns
    from scipy.stats import f_oneway as anova
    from scipy.stats import kruskal as kwtest
    import numpy as np
    
    # load data and format to useful object
    samples = []
    HM_files = os.listdir(HM_outdir)
    for HM_data in HM_files:
        HM_df = pd.read_csv(HM_outdir + HM_data)
        HM_df['label'] = HM_data[6:-4]
        samples.append(HM_df[HM_df.avg_speed>speed_thresh])
    Hist_df = pd.concat(samples, ignore_index = True)
    Hist_df = Hist_df.drop(['Well/XY'], axis = 1).drop(['cell_id'], axis = 1)

    # generate list of parameters to analyze    
    paralist = [p for p in Hist_df.columns if p != 'label']
#    paralist = ['total_distance','avg_moving_speed01' , 'linearity']  #testlist
#    paralist = [p for p in Hist_df.columns if p.startswith('avg_moving_speed')]  #testlist
    print(str(len(paralist)) + ' boxplots will be created\nfinished boxplots:',end=' ')

    # create graphs for each parameter
    for i,para in enumerate(paralist):
        plt.figure(para)
        
        # ignore 0 values in avg_moving_speed
        if para.startswith('avg_moving_speed'):
            Hist_df[para] = [x if x>0 else np.nan for x in Hist_df[para]]
            
        # generate boxplot with overlying datapoints
        ax = sns.boxplot    (x='label', y=para, data=Hist_df, showfliers=False)
        ax = sns.swarmplot  (x='label', y=para, data=Hist_df, color='black', alpha=0.5)
        # graph formatting
        ax.set_ylim(ax.get_ylim()[0],ax.get_ylim()[1]*1.075)
        ax.xaxis.label.set_visible(False)
        
        # perform comparitive statistics on each plot and state them above plot
        Anov_F,Anov_p   = anova (*[list(g[para]) for g in samples])
        kw_H,kw_p       = kwtest(*[list(g[para]) for g in samples])
        plt.title(f'ANOVA p={round(Anov_p,p_dec)}; Kruskal-Wallis p={round(kw_p,p_dec)}')

        # get mean, standard deviation, and N for each graph and state it above bar
        groupmeans = Hist_df.groupby(['label'])[para].mean().values
        groupstdvs = Hist_df.groupby(['label'])[para].std().values
        groupsizes = Hist_df.groupby(['label'])[para].count().values

        pos = range(len(groupmeans))
#        vpos = 
        for tick,label in zip(pos,ax.get_xticklabels()):
            ax.text(pos[tick], 0.99*ax.get_ylim()[1],
                    f'{np.round(groupmeans[tick],2)}\n+/- {np.round(groupstdvs[tick],2)}\nN = {groupsizes[tick]}',
                    ha='center', va = 'top', size='x-small', weight='semibold', color=sns.color_palette()[tick])
        
        # save and close figure
        plt.savefig(Quant_analys + str(i+1) + '_' + para+'.png',dpi=1200)
        plt.close()

        print (str(i+1),end=', ')

    CPfinish = datetime.now()
    CPtime = int(round((CPfinish - CPstart).total_seconds(),0))
    print ('boxplots all done after %i seconds'%CPtime)
    print ('')

#%%         
finishtime = datetime.now()
script_dur = int(round((finishtime - starttime).total_seconds(),0))
print ('script total duration: %i seconds'%script_dur)