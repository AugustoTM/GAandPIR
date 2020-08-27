'''
Created on 29 mai 2018

@author: t0194071
'''

############################################################################################################################################
'''                                                       IMPORT AND FUNCTIONS DECLARATION                                               '''
############################################################################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
import json
import pickle
import numpy as np
 
def getInterval(data,timestamp):
    for index, element in enumerate(data["Timestamp"]):
        if element>timestamp[0]:
            b=index
            break
    for index, element in enumerate((data["Timestamp"])):
        if element>timestamp[1]:
            e=index
            break
    return b,e   

def windowExtraction(data,size,list):
    for i in range(0,len(data)-size,round(size/2)):
        list.append(data[i:size+i])
    return list

def getcolor(level):
    if level=="L":
        return "green"
    elif level=="M":
        return "orange"
    elif level=="H":
        return "red"
    else:
        return "black"
    
def getSimplifiedCoordinate(coord,axis):
    
    if axis=='X':
        scale=[0,18,50,70,105,120,150]
        Scoord=[5,-3,-2,-1,1,2,3,5]
    else: 
        scale=[-20,8,55,100]
        Scoord=[5,1,0,-1,5]
    pos=0
    if coord<scale[0]:
        pos=0
    elif coord>scale[-1]:
        pos=len(Scoord)-1
    else:
        for i in range(0,len(scale)-1):
            if coord>scale[i] and coord<scale[i+1]:
                pos=i+1
                break
    return Scoord[pos]
    
def getAOI(coords):
    coords=(int(coords[0]),int(coords[1]))
    if coords==(-1,1):
        return 0
    elif coords==(1,1):
        return 1
    elif coords==(-1,0) or coords==(1,0):
        return 2
    elif coords==(-2,0):
        return 3
    elif coords==(-2,-1):
        return 4
    elif coords==(-1,-1) or coords==(1,-1):
        return 5
    elif coords==(3,0) or coords==(3,-1):
        return 6
    elif coords==(-3,1) or coords==(-3,0) or coords==(-3,-1):
        return 7
    else:
        return 8
    
def plotSignal(df,signalFeature):
    plt.Figure
    ymax=max(df[signalFeature])
    ymin=min(df[signalFeature])
    df.plot(x="Timestamp", y=signalFeature)
    plt.title(signalFeature)
    plt.show()
    
def plotSignalAOI(df,signalFeature):
    plt.Figure
    ymax=max(df[signalFeature])
    ymin=min(df[signalFeature])
    df.plot(x="Timestamp", y=signalFeature)
    plt.title(signalFeature)
    plt.yticks(np.arange(9), ('Points','Chrono','TRACKING','SYSMON','COMM','RESMON','TUNN','MainScore','Dehors'))
    plt.show()
    
def writeToJSONFile(path, fileName, data):
    filePathNameWExt = './' + path + '/' + fileName + '.json'
    with open(filePathNameWExt, 'w') as fp:
        json.dump(data, fp)
        
def getColor(number):
    colors=['yellow','orange','red','green','blue','brown','violet','grey','black']
    return colors[int(number)]

############################################################################################################################################
'''                                                               PARAMETERS                                                             '''
############################################################################################################################################
    
subjectName='T2'

############################################################################################################################################
'''                                                            SCRIPT PARAMETERS                                                         '''
############################################################################################################################################        

dataToSelectVT3Raw=['LeftEye_GazePoint_X', 'LeftEye_GazePoint_Y','SensorHubTimestampInMillisecond']
signalFeatures=['LeftEye_GazePoint_X', 'LeftEye_GazePoint_Y','Area_X','Area_Y']
#levels=["L1","M1","H1","L2","M2","H2"]
levels=["H1","L2","M2","H2"]

#path=os.path.join("F:\\Data\\Users\\T0194071\\Files",subjectName)
path=os.path.join("C:/Users/atmiy/Documents/ISAE SUPAERO/2A/S4/PIR/Data/AOI",subjectName)
savepath=os.path.join(path,'ProcessedData')
datapath=os.path.join(path,'RawData')

############################################################################################################################################
'''                                                    Data extraction and AOI computation                                               '''
############################################################################################################################################        


for file in os.listdir(savepath):
    if "BlockSensorHubDict" in file:
        summaryName=file
        break
with open(os.path.join(savepath,summaryName), 'rb') as pickle_file:
    dataBlock=pickle.load(pickle_file)
    
dfVT3AOI=[]
for level in levels:
    dfVT3Raw=dataBlock[level]
    
    for i, row in dfVT3Raw.iterrows():
        dfVT3Raw.at[i,'Area_X']=getSimplifiedCoordinate(dfVT3Raw['LeftEye_GazePoint_X'][i],'X')
        dfVT3Raw.at[i,'Area_Y']=getSimplifiedCoordinate(dfVT3Raw['LeftEye_GazePoint_Y'][i],'Y')
        dfVT3Raw.at[i,'AOI']=getAOI((dfVT3Raw['Area_X'][i],dfVT3Raw['Area_Y'][i]))
    
#    plotSignalAOI(dfVT3Raw,'AOI')
     
    im = plt.imread('F:\\Data\\Users\\T0194071\\Files\\Common\\Screenshot.png')
    plt.imshow(im,zorder=1)
    for i, row in dfVT3Raw.iterrows():
        plt.scatter(dfVT3Raw['LeftEye_GazePoint_X'][i]*(1920/150),dfVT3Raw['LeftEye_GazePoint_Y'][i]*(1080/100),s=10,zorder=2, color=getColor(dfVT3Raw['AOI'][i]))
    plt.show()
    
    dfVT3AOI.append(dfVT3Raw)
    
    
############################################################################################################################################
'''                                                                Ratios                                                               '''
############################################################################################################################################ 
AOIfetures={} 
for level in levels:
    dfVT3Raw=dataBlock[level]
    AOIfetures[level]=((len(dfVT3Raw.loc[dfVT3Raw['AOI'] == 2])+len(dfVT3Raw.loc[dfVT3Raw['AOI'] == 1])+len(dfVT3Raw.loc[dfVT3Raw['AOI'] == 0]))/len(dfVT3Raw))
  
############################################################################################################################################
'''                                                               Save data                                                              '''
############################################################################################################################################   

with open(os.path.join(savepath,subjectName+"_BlockAOIDict.pkl"),"wb") as f:
    pickle.dump(dfVT3AOI, f)
    
with open(os.path.join(savepath,subjectName+"_AOIFeatures.pkl"),"wb") as f:
    pickle.dump(AOIfetures, f)
