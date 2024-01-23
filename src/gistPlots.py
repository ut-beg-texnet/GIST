"""
gistPlots.py

Generate plots for GIST results
Dependencies:
    numpy, pandas, matplotlib, seaborn, geopandas, contextily, folium, mapclassify
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas
import contextily as cx
from mpl_toolkits.axes_grid1 import make_axes_locatable


def histogramMC(shallowM,deepM):
  """
  histogramMC - plot histograms of Monte Carlo parameters
  Inputs: shallowM - GIST shallow instance
          deepM    - GIST deep instance
  """
  # Visualizations of parameter ranges
  fig,axes=plt.subplots(12,2, sharey=True, figsize=(10,15))
  sns.histplot(ax=axes[0,0],x=shallowM.rhoVec)
  sns.histplot(ax=axes[0,1],x=deepM.rhoVec)
  axes[0,1].set_title('Deep')
  axes[0,0].set_title('Shallow')
  axes[0,0].set_xlabel('Fluid Density (kg/m3)')
  axes[0,1].set_xlabel('Fluid Density (kg/m3)')
  sns.histplot(ax=axes[1,0],x=shallowM.ntaVec)
  sns.histplot(ax=axes[1,1],x=deepM.ntaVec)
  axes[1,0].set_xlabel('Fluid Viscosity (Pa s)')
  axes[1,1].set_xlabel('Fluid Viscosity (Pa s)')
  sns.histplot(ax=axes[2,0],x=shallowM.phiVec)
  sns.histplot(ax=axes[2,1],x=deepM.phiVec)
  axes[2,1].set_xlabel('Porosity (percent)')
  axes[2,0].set_xlabel('Porosity (percent)')
  sns.histplot(ax=axes[3,0],x=shallowM.kMDVec)
  sns.histplot(ax=axes[3,1],x=deepM.kMDVec)
  axes[3,1].set_xlabel('Permeability (mD)')
  axes[3,0].set_xlabel('Permeability (mD)')
  sns.histplot(ax=axes[4,0],x=shallowM.hVec)
  sns.histplot(ax=axes[4,1],x=deepM.hVec)
  axes[4,0].set_xlabel('Thickness (ft)')
  axes[4,1].set_xlabel('Thickness (ft)')
  sns.histplot(ax=axes[5,0],x=shallowM.alphavVec)
  sns.histplot(ax=axes[5,1],x=deepM.alphavVec)
  axes[5,1].set_xlabel('Vertical Compressibility (1/Pa)')
  axes[5,0].set_xlabel('Vertical Compressibility (1/Pa)')
  sns.histplot(ax=axes[6,0],x=shallowM.betaVec)
  sns.histplot(ax=axes[6,1],x=deepM.betaVec)
  axes[6,1].set_xlabel('Fluid Compressibility (1/Pa)')
  axes[6,0].set_xlabel('Fluid Compressibility (1/Pa)')
  sns.histplot(ax=axes[7,0],x=shallowM.muVec)
  sns.histplot(ax=axes[7,1],x=deepM.muVec)
  axes[7,1].set_xlabel('Shear Modulus (Pa)')
  axes[7,0].set_xlabel('Shear Modulus (Pa)')
  sns.histplot(ax=axes[8,0],x=shallowM.nuVec)
  sns.histplot(ax=axes[8,1],x=deepM.nuVec)
  axes[8,1].set_xlabel("Drained Poisson's Ratio (-)")
  axes[8,0].set_xlabel("Drained Poisson's Ratio (-)")
  sns.histplot(ax=axes[9,0],x=shallowM.nu_uVec)
  sns.histplot(ax=axes[9,1],x=shallowM.nu_uVec)
  axes[9,1].set_xlabel("Undrained Poisson's Ratio (-)")
  axes[9,0].set_xlabel("Undrained Poisson's Ratio (-)")
  sns.histplot(ax=axes[10,0],x=deepM.alphaVec)
  sns.histplot(ax=axes[10,1],x=shallowM.alphaVec)
  axes[10,1].set_xlabel("Biot's Coefficient (-)")
  axes[10,0].set_xlabel("Biot's Coefficient (-)")
  sns.histplot(ax=axes[11,0],x=shallowM.diffPPVec)
  sns.histplot(ax=axes[11,1],x=deepM.diffPPVec)
  axes[11,1].set_xlabel("Diffusivity (m2/s)")
  axes[11,0].set_xlabel("Diffusivity (m2/s)")
  plt.tight_layout(pad=1.2)
  plt.show()

def rMinusTPlot(ppWellDF,peWellDF,minYear=-40,sizeTuple=(10,300),title='Well Selection'):
  """
  rMinusTPlot

  Generate plot of injection prior to an earthquake, with well selections.

  Inputs:
      ppWellDF    Dataframe of well selection for pore pressure case from gistMCLive
      peWellDF    Dataframe of well selection for poroelastic case from gistMCLiv
      minYear     How many years before the earthquake to plot
      sizeTuple   Tuple (,) of smallest and largest dots for wells sized by volume
      title       Text for title of plot
"""
  wellDF=ppWellDF.copy()
  wellDF['MMBBL']=wellDF['TotalBBL']/1000000.
  wellDF['YearsInjectingToEarthquake']=-wellDF['YearsInjecting']
  peWellDF['Poroelastic']=peWellDF['Selected']
  wellDF=wellDF.merge(peWellDF[['ID','Poroelastic']],on='ID')
  wellDF['Selection']='None'
  wellDF.loc[wellDF['Poroelastic'],'Selection']='Poroelastic'
  wellDF.loc[wellDF['Selected'],'Selection']='Pore Pressure'
  fig, ax = plt.subplots(figsize=(18,12))
  plt.title(title)
  sns.scatterplot(data=wellDF,x='YearsInjectingToEarthquake', y='Distances',size='MMBBL',hue='Selection',palette={"None": "k", "Poroelastic": "m", "Pore Pressure": "g"},legend='auto',sizes=(10,300), ax=ax)
  ax.set_xlabel('Years Since Earthquake')
  ax.set_ylabel('Distance From Earthquake (km)')
  sns.move_legend(ax, "upper left")
  plt.xlim((minYear,0))

def allWellMaps(eq,deepPPWells,shallowPPWells,deepPEWells,shallowPEWells,dday):
  fig,axes=plt.subplots(2,2, sharey='row', sharex='row', figsize=(35,25))
  wellMap(axes[0,0],eq,deepPPWells,'Deep','Pore Pressure',dday)
  wellMap(axes[0,1],eq,shallowPPWells,'Shallow','Pore Pressure',dday)
  wellMap(axes[1,0],eq,deepPEWells,'Deep','Poroelastic',dday)
  wellMap(axes[1,1],eq,shallowPEWells,'Deep','Pore Pressure',dday)
  plt.show()

# wellMap needs three things - pore pressure wells (with labels), poroelastic wells (no labels), earthquake

# wellMaps should make one for shallow and one for deep
def intervalWellMap(eventID,interval,PEWells,PPWells,eq):
  fig, ax = plt.subplots(figsize=(24,24))
  plt.title('Earthquake '+eventID+' '+interval+' wells')
  divider = make_axes_locatable(ax)
  PPWellsGDF = geopandas.GeoDataFrame(PPWells, geometry=geopandas.points_from_xy(PPWells['SurfaceHoleLongitude'], PPWells['SurfaceHoleLatitude']), crs="EPSG:4326")
  PEWellsGDF = geopandas.GeoDataFrame(PEWells, geometry=geopandas.points_from_xy(PEWells['SurfaceHoleLongitude'], PEWells['SurfaceHoleLatitude']), crs="EPSG:4326")
  xlim=(min(PEWellsGDF['SurfaceHoleLongitude'])-0.1,max(PEWellsGDF['SurfaceHoleLongitude'])+0.1)
  ylim=(min(PEWellsGDF['SurfaceHoleLatitude'])-0.1,max(PEWellsGDF['SurfaceHoleLatitude'])+0.1)
  #PEWellsGDF.plot(ax=ax,color='blue')
  #PPWellsGDF.plot(ax=ax,color='red')
  PEWellsGDF.plot(ax=ax,color='DDRatio')
  eq.plot(ax=ax,color='purple',marker='*',markersize=300)
  cx.add_basemap(ax,zoom=10,crs="EPSG:4326")
  # Legend
  import matplotlib.lines as mlines
  blueCircle = mlines.Line2D([], [], color='blue', marker='.', linestyle='None',
                          markersize=20, label=interval+' Poroelastic Wells')
  redCircle = mlines.Line2D([], [], color='red', marker='x', linestyle='None',
                          markersize=20, label=interval+' Pore Pressure Wells')
  purpleStar = mlines.Line2D([], [], color='purple', marker='*', linestyle='None',
                          markersize=20, label=eventID)
  plt.legend(handles=[blueCircle, redCircle, purpleStar])
  plt.show()

def wellMap(ax,earthquake,wells,inj,zone,physics,dday):
  # Merge well names onto injection
  injQC=inj.merge(wells,on='ID')
  injQC['Date']=pd.to_datetime(injQC.Date)
  wells['TotalBBL']=inj.groupby('ID',as_index=False)['BPD'].sum()['BPD']*dday
  earthquake['TotalBBL']=100000000.
  wellsAndEQ=pd.concat([wells.rename(columns={"SurfaceHoleLongitude":"Longitude","SurfaceHoleLatitude":"Latitude","WellName":"Name"}).assign(dataset=zone+'Wells'),earthquake.rename(columns={"Longitude (WGS84)":"Longitude","Latitude (WGS84)":"Latitude","EventId":"Name"}).assign(dataset='EQ')])
  # Plot
  sns.scatterplot(ax=ax,x='Longitude',y='Latitude',hue='Name',style="dataset",data=wellsAndEQ,size='TotalBBL',sizes=(50.,500.))
  ax.set_title('Selected '+zone+' Wells ('+physics+')')
  ax.set_xlabel('Longitude (deg)')
  ax.set_ylabel('Latitude (deg)')

# wellInjPlot should make a stacked area plot of injection from selected wells - specify time interval, labels
def stackInjPlot(injDF,wellDF,wellList,startDay,dD):
  # Get subset of well names, injection
  fWellDF=wellDF[wellDF['ID'].isin(wellList)]
  fInjDF=injDF[injDF['ID'].isin(wellList)]
  nWellsIn=len(wellList)
  nWellsFound=wellDF.shape[0]
  if nWellsIn != nWellsFound: print(nWellsIn-nWellsFound,' wells not found')
  maxDay=fInjDF['Days'].max()
  nD=1+ (maxDay-startDay)/dD
  injArray=np.zeros([nD,nWellsFound])
  nameList=[]
  i1=0
  for well in wellList:
    inj=fInjDF[fInjDF['ID']==well]
    oD=min(inj.Days)
    injNP=inj.BPD.to_numpy()
    i0=(oD-startDay)/dD
    e0=(max(inj.Days)-startDay)/dD
    # I do not check the end date, assume they are all the same
    if i0<=0: # This means the start Day is after the first injection - 
      injArray[:e0,i1]=injNP[i0:]
    else:
      injArray[i0:,i1]=injNP[:]
    nameList.append(fWellDF['Name'][fWellDF['ID']==well])
    i1=i1+1
  fig, ax = plt.subplots(figsize=(12,24))
  ax.stackplot(data=injArray,labels=nameList)
  ax.set_title('Injection')
  ax.set_xlabel('Days')
  ax.set_ylabel('Rate (BPD)')
  
  # plot stackplot
# we need plots of individual wells also - put both deep and shallow listings of wells to see what is going on

def wellInjPlot(injDF,wellDF,startDate,endDate,title,legend):
  fig, ax = plt.subplots(figsize=(12,24))
  # Merge well information onto injection rates
  # Need to incorporate time sampling other than 10 day
  injQC=injDF.merge(wellDF,on='ID')
  injQC['Date']=pd.to_datetime(injQC.Date)
  # Generate x and y
  dates=injQC['Date'].drop_duplicates()
  # Loop over wells
  oDate=min(dates)
  eDate=max(dates)
  dateRange=pd.date_range(start=oDate,end=eDate, freq='10D')
  # Get x min, x max
  wellIDs=injQC['ID'].drop_duplicates()
  allWellBPDs=[]
  allWellLabels=[]
  for well in wellIDs:
    wellDates=injQC[injQC['ID']==well]['Date']
    wellBPDs=injQC[injQC['ID']==well]['BPD']
    minWD=min(wellDates)
    maxWD=max(wellDates)
    nPrepend=sum(dateRange<minWD)
    nAppend=sum(dateRange>maxWD)
    #print('Well ',well,' ',minWD,maxWD,oDate,eDate,nPrepend,' prePad, ',nAppend,' postPad, ',len(dateRange))
    #print('allWellBPDs: ',allWellBPDs)
    if nPrepend>0:
      prependZeroBPDs=pd.Series(np.zeros([nPrepend]))
    else:
      prependZeroBPDs=None
    if nAppend>0:
      appendZeroBPDs=pd.Series(np.zeros([nAppend,1]))
    else:
      appendZeroBPDs=None
    padWellBPDs=pd.concat([prependZeroBPDs,wellBPDs,appendZeroBPDs])
    allWellBPDs.append(padWellBPDs)
    #allWellLabels.append(injQC['WellName'][injQC['ID']==well][0])
    allWellLabels.append(injQC['WellName'][injQC['ID']==well].loc[0])
  # Now have allWellBPDs, allWellLabels, and dateRange all set
  # Generate Stacked area plot of injection
  sns.set_theme()
  # y needs to be defined for each well, and extended for the full time series
  col = sns.color_palette("hls", len(wellIDs))
  #zoomDate=oneEQ['Origin Date'] - pd.DateOffset(years=2)
  ax.stackplot(dateRange,allWellBPDs, labels=allWellLabels,colors=col)
  ax.set_title(title)
  ax.set_xlabel('Date')
  ax.set_ylabel('Rate (BPD)')
  #if startDate!=None or endDate!=None: axes.xlim(startDate,endDate)
  if legend: ax.legend(loc='upper left')
  plt.show()

def disaggregationPlot(ax,pp,pe,wells,title):
  pp['Physics']='Pore Pressure'
  pe['Physics']='Poroelastic'
  pp['Stresses']=pp['Pressures']
  both=pd.concat(objs=[pp,pe[pe['ID'].isin(pp['ID'])]])
  both=both.merge(wells,on='ID')
  # Relative contributions
  sns.stripplot(data=both,x='Stresses',y='WellName',hue='Physics',dodge=True,jitter=True,alpha=0.1,ax=ax)
  ax.set_title(title+' Well Stresses')
  ax.set_xlabel('Stress (PSI)')
