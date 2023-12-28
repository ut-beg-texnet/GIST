import scipy.special as sc
from scipy import interpolate
import numpy as np
import pandas as pd
import math
#
# injectionV3.py
# bill.curry@exxonmobil.com
#

#############################################
# Classes to process injection data from B3 #
#############################################
# Contains:
#    Classes:
#      injTX - Texas v3 injection data
#         init
#         addMonthly
#         addDaily
#      injNM - New Mexico v3 injection data
#         init
#         addMonthly
#         addDaily
#      inj   - merged injection data
#         init
#         processRates
#         output
#    Subroutines:
#       filterBPDs
#       getDays
#       regularize
#       writeDictDF
#############################################

class injTX:
  ####################################################################
  # Class to read and parse portions of B3 Texas injection data (v3) #
  #   To-do: point to a REST API for inputs instead of files         #
  ####################################################################
  def __init__(self,InjectionWellFile,deepShallow,depth,OutFile=None,verbose=0):
    #########################################
    # Initialize Texas injection data class #
    #########################################
    # Arguments #
    # InjectionWellFile - Full path/name of B3 "InjectionWell.csv" file
    # deepShallow       - Assuming two disposal intervals: "Deep" or "Shallow" - used to filter B3 wells
    # depth             - Depth cutoff between Deep and Shallow wells for wells without a B3 classification
    # OutFile           - Full path/name of windowed output well file
    ###########################################################
    # List of columns that we want to keep from InjectionWell #
    ###########################################################
    keepColumns=['InjectionWellId','APINumber','UICNumber','Basin','SurfaceHoleLatitude','SurfaceHoleLongitude','WellName','CompletedWellDepthClassification','InjectionType','InjectionStatus','RRCInjectionStatus','WellActivatedDate','PermittedMaxLiquidBPD','PermittedIntervalBottomFt','PermittedIntervalTopFt','IsAuthorizedToInjectSaltWater']

    ############################################
    # Read Injection Well file - list of wells #
    ############################################
    TempWellDF=pd.read_csv(InjectionWellFile,usecols=keepColumns,low_memory=False,index_col=False)
    if verbose>0: print(' injectionV3.injTX: Injection Well loaded with ',TempWellDF.shape[0],' wells')

    ########################
    # Filtering operations #
    ##########################
    # Limit Basin to Permian #
    ##########################
    TempWellPermianDF=TempWellDF[(TempWellDF['Basin'].str.contains('Permian',case=False,na=False))]
    if verbose>0: print(' injectionV3.injTX: Basin filtered to ',TempWellPermianDF.shape[0],' wells')
    ###############################################################
    # Limit InjectionType to Injection Into a Non-Productive Zone #
    ###############################################################
    TempWellFiltDF=TempWellPermianDF[(TempWellPermianDF['InjectionType']=='Injection Into Non-Productive Zone')]
    if verbose>0: print(' injectionV3.injTX: InjectionType filtered to ',TempWellFiltDF.shape[0],' wells')
    ######################################################
    # I did not filter to active wells, seems unreliable #
    ######################################################
    #########################################################
    # Filter based on permitted depth of injection interval #
    #########################################################
    if deepShallow=='Deep':
      self.wellDepthDF=TempWellFiltDF[(TempWellFiltDF['PermittedIntervalTopFt']>depth)]
    elif deepShallow=='Shallow':
      self.wellDepthDF=TempWellFiltDF[(TempWellFiltDF['PermittedIntervalBottomFt']<depth)]
    if verbose>0: print(' injectionV3.injTX: PermittedInterval filtered to ',self.wellDepthDF.shape[0],' wells')
    ############################################################################################
    # Secondary filter on B3's CompletedWellDepthClassification based on an incomplete horizon #
    ############################################################################################
    self.wellClassDF=TempWellFiltDF[(TempWellFiltDF['CompletedWellDepthClassification'].str.contains(deepShallow,case=False,na=False))]
    if verbose>0: print(' injectionV3.injTX: CompletedWellDepthClassification filtered to ',self.wellClassDF.shape[0],' wells')
    ######################################
    # Merge both depth filtering results #
    ######################################
    self.wellDF=pd.concat([self.wellClassDF,self.wellDepthDF])
    self.wellDF.drop_duplicates(subset=['InjectionWellId'], keep='first', inplace=True, ignore_index=True)
    #############################################
    # Rename 'WellActivatedDate' to 'StartDate' #
    #############################################
    self.wellDF=self.wellDF.rename(columns={'WellActivatedDate':'StartDate'})
    if verbose>0: print(' injectionV3.injTX: Merged list is ',self.wellDF.shape[0],' wells')
    ##########################################################
    # Write out filtered list of wells if OutFile is present #
    ##########################################################
    if OutFile!=None: self.wellDF.to_csv(OutFile,index=False)
    #################################
    # Store list of wells in object #
    #################################
    self.wellList=self.wellDF['InjectionWellId'].tolist()
    ##########################################
    # Set columns for time series dictionary #
    ##########################################
    self.timeSeriesColumns=['InjectionWellId','Date','BPD','Modeled']
    #########################
    # Initialize dictionary #
    #########################
    self.timeSeriesDict=dict()
    ########################################################
    # Initialize one dataframe per dictionary entry (well) #
    ########################################################
    for well in self.wellList:
      self.timeSeriesDict[str(well)]=pd.DataFrame(columns=self.timeSeriesColumns)
    return


  def addMonthly(self,MonthlyWellFile,chunkSize,OutFile=None,verbose=0):
    #############
    # Arguments #############################################################################
    # MonthlyWellFile - Full path/name of B3 "MonthlyInjection.csv" file                    #
    # chunkSize       - Number of rows of file to read                                      #
    # OutFile         - Full path/name of windowed output monthly injection file (optional) #
    #########################################################################################
    # Read Monthly well file - can be too large for memory so read in chunks #
    ##########################################################################
    iterCSV=pd.read_csv(MonthlyWellFile,low_memory=False,iterator=True,chunksize=chunkSize)
    if verbose>0: print(' injectionV3.injTX.addMonthly: ',len(self.wellList),' wells')
    ######################################################
    # Initialize chunk number and count of filtered rows #
    ######################################################
    chunk=0
    rowCount=0
    wList=[]
    #####################
    # Iterate over file #
    #####################
    for CSVchunk in iterCSV:
      #####################################################
      # Get list of unique InjectionWellIds in this chunk #
      #####################################################
      chunkWellList=set(CSVchunk['InjectionWellId'].tolist())
      ################################################################################################
      # Loop over chunkWellList, if well is in self.wellList then append rows to self.timeSeriesDict #
      ################################################################################################
      if verbose>0: print(' injectionV3.injTX.addMonthly: ',len(chunkWellList),' wells in chunk ',chunk,' of file')
      for well in chunkWellList:
        ################################
        # Check if well is in our list #
        ################################
        if well in self.wellList:
          wList=wList+[well] # Append a list to a list
          ###########################################
          # Select rows for the well from the chunk #
          ###########################################
          wellDF=CSVchunk[CSVchunk['InjectionWellId']==well]
          rowCount=rowCount+wellDF.shape[0]
          if verbose>1: print(' injectionV3.injTX.addMonthly: ',wellDF.shape[0],' rows found for well ',well)
          ##################################################################
          # Select the columns and change their names to timeSeriesColumns #
          ##################################################################
          wellWinDF=wellDF[['InjectionWellId','StartOfMonthDate','InjectedLiquidBPD','InjectedLiquidIsModeled']]
          wellWinDF.columns=self.timeSeriesColumns
          #############################################
          # Append to the dataframe in the dictionary #
          #############################################
          self.timeSeriesDict[str(well)]=self.timeSeriesDict[str(well)].append(wellWinDF)
          if verbose>1: print(' injectionV3.injTX.addMonthly: ',self.timeSeriesDict[str(well)].shape[0],' rows total for well ',well)
        # end if
      chunk=chunk+1
    # end for
    injWellList=set(wList)
    if verbose>0: print(' injectionV3.injTX.addMonthly: ',len(injWellList),' wells added from file with ',rowCount,' monthly injections')
    ##############################################################
    # if output file is specified, write out self.timeSeriesDict #
    ##############################################################
    if OutFile!=None: writeDictDF(self.timeSeriesDict,OutFile)
    return
  
  def addDaily(self,DailyWellFile,chunkSize,OutFile=None,verbose=0):
    #############
    # Arguments ###########################################################################
    # DailyWellFile - Full path/name of B3 "DailyInjection.csv" file                      #
    # chunkSize       - Number of rows of file to read                                    #
    # OutFile         - Full path/name of windowed output daily injection file (optional) #
    #######################################################################################
    # Read Daily well file - can be too large for memory so read in chunks #
    ########################################################################
    iterCSV=pd.read_csv(DailyWellFile,low_memory=False,iterator=True,chunksize=chunkSize)
    if verbose>0: print(' injectionV3.injTX.addDaily: ',len(self.wellList),' wells')
    ######################################################
    # Initialize chunk number and count of filtered rows #
    ######################################################
    chunk=0
    rowCount=0
    wList=[]
    #####################
    # Iterate over file #
    #####################
    for CSVchunk in iterCSV:
      #####################################################
      # Get list of unique InjectionWellIds in this chunk #
      #####################################################
      chunkWellList=set(CSVchunk['InjectionWellId'].tolist())
      ################################################################################################
      # Loop over chunkWellList, if well is in self.wellList then append rows to self.timeSeriesDict #
      ################################################################################################
      if verbose>0: print(' injectionV3.injTX.addDaily: ',len(chunkWellList),' wells in chunk ',chunk,' of file')
      for well in chunkWellList:
        ################################
        # Check if well is in our list #
        ################################
        if well in self.wellList:
          wList=wList+[well] # Append a list to a list
          ###########################################
          # Select rows for the well from the chunk #
          ###########################################
          wellDF=CSVchunk[CSVchunk['InjectionWellId']==well]
          rowCount=rowCount+wellDF.shape[0]
          if verbose>1: print(' injectionV3.injTX.addDaily: ',wellDF.shape[0],' rows found for well ',well)
          ##################################################################
          # Select the columns and change their names to timeSeriesColumns #
          ##################################################################
          wellWinDF=wellDF[['InjectionWellId','Date','InjectedLiquidBBL','InjectedLiquidIsModeled']]
          wellWinDF.columns=self.timeSeriesColumns
          #############################################
          # Append to the dataframe in the dictionary #
          #############################################
          self.timeSeriesDict[str(well)]=self.timeSeriesDict[str(well)].append(wellWinDF)
          if verbose>1: print(' injectionV3.injTX.addDaily: ',self.timeSeriesDict[str(well)].shape[0],' rows total for well ',well)
        # end if
      chunk=chunk+1
    # end for
    self.injWellList=set(wList)
    if verbose>0: print(' injectionV3.injTX.addDaily: ',len(self.injWellList),' wells added from file with ',rowCount,' daily injections')
    ##############################################################
    # if output file is specified, write out self.timeSeriesDict #
    ##############################################################
    if OutFile!=None: writeDictDF(self.timeSeriesDict,OutFile)
    return
  
class injNM:
  ####################################################################
  # Class to read and parse portions of B3 Texas injection data (v3) #
  #   To-do: point to a REST API for inputs instead of files         #
  ####################################################################
  def __init__(self,InjectionWellFile,deepShallow,depth,OutFile=None,verbose=0):
    ##############################################
    # Initialize New Mexico injection data class #
    ##############################################
    # Arguments #############################################################################################
    # InjectionWellFile - Full path/name of B3 "InjectionWell.csv" file                                     #
    # deepShallow       - Assuming two disposal intervals: "Deep" or "Shallow" - used to filter B3 wells    #
    # depth             - Depth cutoff between Deep and Shallow wells for wells without a B3 classification #
    # OutFile           - Full path/name of windowed output well file (optional)                            #
    #########################################################################################################
    # List of columns that we want to keep from InjectionWell #
    ###########################################################
    keepColumns=['InjectionWellId','APINumber','UICNumber','Basin','SurfaceHoleLatitude','SurfaceHoleLongitude','WellName','CompletedWellDepthClassification','InjectionType','InjectionStatus','WellStatus','SpudDate','PermittedMaxLiquidBPD','PermittedIntervalBottomFt','PermittedIntervalTopFt']
    ############################################
    # Read Injection Well file - list of wells #
    ############################################
    TempWellDF=pd.read_csv(InjectionWellFile,usecols=keepColumns,low_memory=False,index_col=False)
    if verbose>0: print(' injectionV3.injNM: Injection Well loaded with ',TempWellDF.shape[0],' wells')
    ########################
    # Filtering operations #
    ####################################################################
    # Limit Basin to Permian - here Delaware or Central Basin Platform #
    #   the polygon setting these in B3 might be too restrictive       #
    ####################################################################
    TempWellDelawareDF=TempWellDF[(TempWellDF['Basin'].str.contains('Delaware',case=False,na=False))]
    if verbose>0: print(' injectionV3.injNM: Basin filtered to ',TempWellDelawareDF.shape[0],' Delaware wells')
    TempWellCBPDF=TempWellDF[(TempWellDF['Basin'].str.contains('Central',case=False,na=False))]
    if verbose>0: print(' injectionV3.injNM: Basin filtered to ',TempWellCBPDF.shape[0],' CBP wells')
    TempWellPermianDF=pd.concat([TempWellDelawareDF,TempWellCBPDF])
    if verbose>0: print(' injectionV3.injNM: Basin filtered to ',TempWellPermianDF.shape[0],' Permian wells')
    #############################################
    # Limit InjectionType to Saltwater Disposal #
    ###################################################
    # InjectionType==Injection appears to be also EOR #
    ###################################################
    TempWellFiltDF=TempWellPermianDF[(TempWellPermianDF['InjectionType']=='Saltwater Disposal')]
    if verbose>0: print('injNM: InjectionType filtered to ',TempWellFiltDF.shape[0],' wells')
    ###################################
    # I did not only set active wells #
    ###################################
    #########################################################
    # Filter based on permitted depth of injection interval #
    #########################################################
    if deepShallow=='Deep':
      self.wellDepthDF=TempWellFiltDF[(TempWellFiltDF['PermittedIntervalTopFt']>depth)]
    elif deepShallow=='Shallow':
      self.wellDepthDF=TempWellFiltDF[(TempWellFiltDF['PermittedIntervalBottomFt']<depth)]
    if verbose>0: print(' injectionV3.injNM: PermittedInterval filtered to ',self.wellDepthDF.shape[0],' wells')
    ############################################################################################
    # Secondary filter on B3's CompletedWellDepthClassification based on an incomplete horizon #
    ############################################################################################
    self.wellClassDF=TempWellFiltDF[(TempWellFiltDF['CompletedWellDepthClassification'].str.contains(deepShallow,case=False,na=False))]
    if verbose>0: print(' injectionV3.injNM: CompletedWellDepthClassification filtered to ',self.wellClassDF.shape[0],' wells')
    ######################################
    # Merge both depth filtering results #
    ######################################
    self.wellDF=pd.concat([self.wellClassDF,self.wellDepthDF])
    self.wellDF.drop_duplicates(subset=['InjectionWellId'], keep='first', inplace=True, ignore_index=True)
    ####################################
    # Rename 'SpudDate' to 'StartDate' #
    ####################################
    self.wellDF=self.wellDF.rename(columns={'SpudDate':'StartDate'})
    if verbose>0: print(' injectionV3.injNM: Merged list is ',self.wellDF.shape[0],' wells')
    ##########################################################
    # Write out filtered list of wells if OutFile is present #
    ##########################################################
    if OutFile!=None: self.wellDF.to_csv(OutFile,index=False)
    #################################
    # Store list of wells in object #
    #################################
    self.wellList=self.wellDF['InjectionWellId'].tolist()
    ##########################################
    # Set columns for time series dictionary #
    ##########################################
    self.timeSeriesColumns=['InjectionWellId','Date','BPD','Modeled']
    #########################
    # Initialize dictionary #
    #########################
    self.timeSeriesDict=dict()
    ########################################################
    # Initialize one dataframe per dictionary entry (well) #
    ########################################################
    for well in self.wellList:
      self.timeSeriesDict[str(well)]=pd.DataFrame(columns=self.timeSeriesColumns)
    return

  def addMonthly(self,MonthlyWellFile,chunkSize,OutFile=None,verbose=0):
    #############
    # Arguments #############################################################################
    # MonthlyWellFile - Full path/name of B3 "MonthlyInjection.csv" file                    #
    # chunkSize       - Number of rows of file to read                                      #
    # OutFile         - Full path/name of windowed output monthly injection file (optional) #
    #########################################################################################
    # Read Monthly well file - can be too large for memory so read in chunks #
    ##########################################################################
    iterCSV=pd.read_csv(MonthlyWellFile,low_memory=False,iterator=True,chunksize=chunkSize)
    if verbose>0: print(' injectionV3.injNM.addMonthly: ',len(self.wellList),' wells')
    ######################################################
    # Initialize chunk number and count of filtered rows #
    ######################################################
    chunk=0
    rowCount=0
    wList=[]
    #####################
    # Iterate over file #
    #####################
    for CSVchunk in iterCSV:
      #####################################################
      # Get list of unique InjectionWellIds in this chunk #
      #####################################################
      chunkWellList=set(CSVchunk['InjectionWellId'].tolist())
      ################################################################################################
      # Loop over chunkWellList, if well is in self.wellList then append rows to self.timeSeriesDict #
      ################################################################################################
      if verbose>0: print(' injectionV3.injNM.addMonthly: ',len(chunkWellList),' wells in chunk ',chunk,' of file')
      for well in chunkWellList:
        ################################
        # Check if well is in our list #
        ################################
        if well in self.wellList:
          wList=wList+[well] # Append a list to a list
          ###########################################
          # Select rows for the well from the chunk #
          ###########################################
          wellDF=CSVchunk[CSVchunk['InjectionWellId']==well]
          rowCount=rowCount+wellDF.shape[0]
          if verbose>1: print(' injectionV3.injNM.addMonthly: ',wellDF.shape[0],' rows found for well ',well)
          ##################################################################
          # Select the columns and change their names to timeSeriesColumns #
          ##################################################################
          wellWinDF=wellDF[['InjectionWellId','StartOfMonthDate','InjectedLiquidBPD','InjectedLiquidIsModeled']]
          wellWinDF.columns=self.timeSeriesColumns
          #############################################
          # Append to the dataframe in the dictionary #
          #############################################
          self.timeSeriesDict[str(well)]=self.timeSeriesDict[str(well)].append(wellWinDF)
          if verbose>1: print(' injectionV3.injNM.addMonthly: ',self.timeSeriesDict[str(well)].shape[0],' rows total for well ',well)
        # end if
      chunk=chunk+1
    # end for
    wellList=set(wList)
    if verbose>0: print(' injectionV3.injNM.addMonthly: ',len(wellList),' wells added from file with ',rowCount,' monthly injections')
    ##############################################################
    # if output file is specified, write out self.timeSeriesDict #
    ##############################################################
    if OutFile!=None: writeDictDF(self.timeSeriesDict,OutFile)
    return

  def addDaily(self,DailyWellFile,chunkSize,OutFile=None,verbose=0):
    #############
    # Arguments ###########################################################################
    # DailyWellFile - Full path/name of B3 "DailyInjection.csv" file                      #
    # chunkSize       - Number of rows of file to read                                    #
    # OutFile         - Full path/name of windowed output daily injection file (optional) #
    #######################################################################################
    # Read Daily well file - can be too large for memory so read in chunks #
    ########################################################################
    iterCSV=pd.read_csv(DailyWellFile,low_memory=False,iterator=True,chunksize=chunkSize)
    if verbose>0: print(' injectionV3.injNM.addDaily: ',len(self.wellList),' wells')
    chunk=0
    rowCount=0
    wList=[]
    #####################
    # Iterate over file #
    #####################
    for CSVchunk in iterCSV:
      #####################################################
      # Get list of unique InjectionWellIds in this chunk #
      #####################################################
      chunkWellList=set(CSVchunk['InjectionWellId'].tolist())
      ################################################################################################
      # Loop over chunkWellList, if well is in self.wellList then append rows to self.timeSeriesDict #
      ################################################################################################
      if verbose>0: print(' injectionV3.injNM.addDaily: ',len(chunkWellList),' wells in chunk ',chunk,' of file')
      for well in chunkWellList:
        ################################
        # Check if well is in our list #
        ################################
        if well in self.wellList:
          wList=wList+[well] # Append a list to a list
          ###########################################
          # Select rows for the well from the chunk #
          ###########################################
          wellDF=CSVchunk[CSVchunk['InjectionWellId']==well]
          rowCount=rowCount+wellDF.shape[0]
          if verbose>1: print(' injectionV3.injNM.addMonthly: ',wellDF.shape[0],' rows found for well ',well)
          ##################################################################
          # Select the columns and change their names to timeSeriesColumns #
          ##################################################################
          wellWinDF=wellDF[['InjectionWellId','Date','InjectedLiquidBBL','InjectedLiquidIsModeled']]
          wellWinDF.columns=self.timeSeriesColumns
          #############################################
          # Append to the dataframe in the dictionary #
          #############################################
          self.timeSeriesDict[str(well)]=self.timeSeriesDict[str(well)].append(wellWinDF)
          if verbose>1: print(' injectionV3.injNM.addMonthly: ',self.timeSeriesDict[str(well)].shape[0],' rows total for well ',well)
        # end if
      chunk=chunk+1
    # end for
    self.injWellList=set(wList)
    if verbose>0: print(' injectionV3.injNM.addDaily: ',len(self.injWellList),' wells added from file with ',rowCount,' monthly injections')
    ##############################################################
    # if output file is specified, write out self.timeSeriesDict #
    ##############################################################
    if OutFile!=None: writeDictDF(self.timeSeriesDict,OutFile)
    return
  
  
class inj:
  ############################################################
  # Class for merged B3 Texas/New Mexico injection data (v3) #
  #   takes injNM and injTX as inputs, processes merged data #
  ############################################################
  def __init__(self,NM,TX,epoch='01-01-1970',OutWellFile=None,OutInjFile=None,verbose=0):
    #############
    # Arguments #
    #############################################################################################################
    # NM          - injNM object - assumed to have wells, monthly & daily volumes, filtered to zone of interest #
    # TX          - injTX object - "                                                #############################
    # epoch       - Date for DayNo=0 - assume Unix epoch unless otherwise specified #
    # OutWellFile - Full path/name of merged well information (optional)            #
    # OutInjFile  - Full path/name of merged injection information (optional)       #
    #################################################################################
    # Set time series output columns #
    ##################################
    self.timeSeriesColumns=['InjectionWellId','Date','BPD','Modeled']
    #############
    # Set epoch #
    #############
    self.epoch=pd.to_datetime(epoch)
    #################################################################
    # Define unique well ID by adding scalars to B3 InjectionWellId #
    #################################################################
    # Adding 1000000 to NM wells and 2000000 to TX wells #
    ######################################################
    NMShift=1000000
    TXShift=2000000
    if verbose>0: print(' injectionV3.inj: NM max ID ',NM.wellDF['InjectionWellId'].max(),"; TX max ID, ",TX.wellDF['InjectionWellId'].max())
    NMDF=NM.wellDF.copy()
    NMDF.insert(0,'ID',NM.wellDF['InjectionWellId']+NMShift)
    TXDF=TX.wellDF.copy()
    TXDF.insert(0,'ID',TX.wellDF['InjectionWellId']+TXShift)
    #########################
    # Merge NM and TX files #
    #########################
    self.wellDF=pd.concat([NMDF,TXDF],join='inner')
    ######################################
    # Optionally output merged well file #
    ######################################
    if OutWellFile!=None: self.wellDF.to_csv(OutWellFile,index=False)
    ##################################################################################
    # Now copy time series dictionaries and change ID numbers so they don't conflict #
    ##################################################################################
    self.allInj={}
    for wellNum in NM.timeSeriesDict:
      self.allInj[str(int(wellNum)+NMShift)]=NM.timeSeriesDict[wellNum]
      self.allInj[str(int(wellNum)+NMShift)]['ID']=NM.timeSeriesDict[wellNum]['InjectionWellId']+NMShift
    for wellNum in TX.timeSeriesDict:
      self.allInj[str(int(wellNum)+TXShift)]=TX.timeSeriesDict[wellNum]
      self.allInj[str(int(wellNum)+TXShift)]['ID']=TX.timeSeriesDict[wellNum]['InjectionWellId']+TXShift
    ###########################################
    # Optionally output merged injection file #
    ###########################################
    if OutInjFile!=None: writeDictDF(self.allInj,OutInjFile)
    return

  def processRates(self,thresh,interval,endDate,keepModeled,verbose=0):
    #####################################################
    # Process merged NM/TX injection rates              #
    # Assume that all wells and rates can fit in memory #
    # Processing includes:                              #
    #          despiking / NaN / empty data removal     #
    #          removing modeled data (optionally)       #
    #          regularization in time                   #
    #####################################################
    #############
    # Arguments #
    #####################################################################################
    # thresh      - threshold to remove spikes in rates - 200000 kbd seems conservative #
    # interval    - number of days between output rates in time                         #
    # endData     - last date to regularize to - will be rounded to interval            #
    # keepModeled - optionally remove modeled rates from P-18s in TX                    #
    #####################################################################################
    # To-do: Check for bad dates (not a Raiders of the Lost Ark reference) #
    ########################################################################
    # Initialize well count #
    #########################
    wellCount=0
    ##################################################################################
    # Initialize dictionaries for kept, spikes, modeled, NaNs, and regularized rates #
    ##################################################################################
    self.keepDict={}
    self.spikeDict={}
    self.modDict={}
    self.nanDict={}
    self.regDict={}
    ###########################################
    # Convert endDay to an integer day number #
    ###########################################
    self.endDate=pd.to_datetime(endDate)
    self.endDay=(self.endDate-self.epoch).days
    if verbose>0: print(' injectionV3.inj.processRates endDate: ',endDate,'; endDay: ',self.endDay)
    #######################
    # Initialize counters #
    #######################
    nKT=0
    nST=0
    nMT=0
    nNT=0
    nSkip=0
    
    if verbose>0: print(" injectionV3.inj.processRates: ",len(self.allInj)," wells")
    #############################################################
    # Loop over wells in dictionary of per-well injection rates #
    #############################################################
    for wellNum in self.allInj:
      ####################################################################
      # Separate valid rates from spikes, (maybe) modeled data, and NaNs #
      ####################################################################
      (wKeepDF,wSpikeDF,wNanDF,wModDF)=filterBPDs(self.allInj[wellNum],thresh,keepModeled)
      nKeep=wKeepDF.shape[0]
      nSpike=wSpikeDF.shape[0]
      nMod=wModDF.shape[0]
      nNan=wNanDF.shape[0]
      nKT=nKT+nKeep
      nST=nST+nSpike
      nMT=nMT+nMod
      nNT=nNT+nNan
      ####################################
      # Check for any valid data (nKeep) #
      ####################################
      if nKeep>0:
        ##############################################
        # Compute column from keepDF with day number #
        ##############################################
        dayNumDF=getDays(wKeepDF,self.epoch)
        #############################
        # Regularize day, bpd pairs #
        #############################
        wRegDF=regularize(dayNumDF,self.endDay,interval,verbose)
        ####################################################
        # Add entries to keep and regularized dictionaries #
        ####################################################
        self.keepDict[wellNum]=dayNumDF
        self.regDict[wellNum]=wRegDF
      else:
        ############################################
        # No valid data, add to skipped well count #
        ############################################
        nSkip=nSkip+1
        if verbose>0: print(' injectionV3.inj.processRates well ',wellNum,' skipped')
      if nSpike>0:
        #############################################
        # Add dictionary entry for well with spikes #
        #############################################
        self.spikeDict[wellNum]=wSpikeDF
      if nMod>0:
        ######################################################
        # Add dictionary entry for well with modeled entries #
        # Note - values also in nKeep when keepModeled=True  #
        ######################################################
        self.modDict[wellNum]=wModDF
      if nNan>0:
        ###########################################
        # Add dictionary entry for well with NaNs #
        ###########################################
        self.nanDict[wellNum]=wNanDF
      if verbose>0: print(" injectionV3.inj.processRates well# ",wellCount,": ",nKeep," kept values, ",nSpike," spikes, ",nMod," modeled values, ",nNan," Nan/Empty values")
      wellCount=wellCount+1
    if verbose>0: print(" injectionV3.inj.processRates totals :",nKT," kept values, ",nST," spikes, ",nMT," modeled values, ",nNT," Nan/Empty values, ",nSkip," wells with no data")
    return
      
  def output(self,prefix,verbose=0):
    ##############################################################
    # I think that I need to output a well listing               #
    # including those that don't have any valid injection values #
    ##############################################################
    writeDictDF(self.regDict,prefix+'Reg.csv',verbose)
    writeDictDF(self.keepDict,prefix+'Filt.csv',verbose)
    writeDictDF(self.spikeDict,prefix+'Spikes.csv',verbose)
    writeDictDF(self.nanDict,prefix+'NaN.csv',verbose)
    writeDictDF(self.modDict,prefix+'Modeled.csv',verbose)

class injRead:
  #
  # Class for reading injection data output from inj class.
  # Implemented in gistMCLive class for now.
  def __init__(self,wellFile,injFile):
    # Open and read well file
    return
    

#######################################
# Generic functions used by inj class #
#######################################
def filterBPDs(wellDF,thresh,keepModeled,verbose=0):
  #################################################
  # filterBPDs - split a data frame into 4 pieces #
  ############################################################################
  # Inputs                                                                   #
  #   wellDF - dataframe for a single well with injection rates              #
  #          - needs columns ['BPD','Modeled','InjectionWellId','ID','Date'] #
  #   thresh - BPD threshold for bad data                                    #
  #   keepModeled - Boolean for splitting modeled data                       #
  ############################################################################
  # Outputs                                                   #
  #   modDF   - Dataframe of modeled values                   #
  #   keepDF  - Dataframe of reasonable values to keep        #
  #   spikeDF - Dataframe of values where BPD>thresh or BPD<0 #
  #   nanDF   - Dataframe of either NaN or empty values       #
  #############################################################
  # Initialize outputs #
  ######################
  wellColumns=wellDF.columns
  modDF=pd.DataFrame(columns=wellColumns)
  keepDF=pd.DataFrame(columns=wellColumns)
  spikeDF=pd.DataFrame(columns=wellColumns)
  nanDF=pd.DataFrame(columns=wellColumns)
  nWSkip=0
  nSpike=0
  nNan=0
  nKeep=0
  ############################################
  # Check number of BPD values for this well #
  ############################################
  nInj=wellDF.shape[0]
  ########################
  # If zero length, skip #
  ########################
  if nInj==0:
    nWSkip=nWSkip+1
  ################
  # if one value #
  ################
  elif nInj==1:
    if verbose>0: print("  inj.filterBPDs: well with one value: ",wellDF)
    ####################
    # Check if a spike #
    ####################
    if (np.isnan(wellDF['BPD'].to_list()[0])):
      nanDF=nanDF.append(well)
      nNan=1
    elif (wellDF['BPD'].to_list()[0]>thresh):
      spikeDF=spikeDF.append(well)
      nSpike=1
    ##################
    # Check if a NaN #
    # If valid append to output data frame #
    ########################################
    else:
      keepDF=keepDF.append(wellDF)
      nKeep=1
  else:
    #############################
    # Now for non-special cases #
    ############################################
    # find masks for different types of values #
    ############################################
    # Spikes, negative values #
    ###########################
    sidx=((wellDF['BPD']>thresh) | (wellDF['BPD']<0.))
    ########
    # NaNs #
    ########
    nidx=pd.isna(wellDF['BPD'])
    ###########
    # Modeled #
    ###########
    midx=(wellDF['Modeled'])
    ######################################
    # Values to keep (including modeled) #
    ######################################
    kidx=np.logical_and(~sidx, ~nidx)
    ###############################
    # Optionally separate modeled #
    ###############################
    if not keepModeled:
      kidx=np.logical_and(kidx, ~midx)
    nSpike=sidx.sum()
    nNan=nidx.sum()
    nKeep=kidx.sum()
    nMod=midx.sum()
    if nSpike+nNan==0 and keepModeled:
      ############################################
      # If all values are good append everything #
      ############################################
      keepDF=keepDF.append(wellDF)
    else:
      #########################################
      # Else split into different data frames #
      #########################################
      if nNan>0:
        nanDF['BPD']=wellDF['BPD'][nidx]
        nanDF['Date']=wellDF['Date'][nidx]
        nanDF['InjectionWellId']=wellDF['InjectionWellId'][nidx]
        nanDF['ID']=wellDF['ID'][nidx]
        nanDF['Modeled']=wellDF['Modeled'][nidx]
      if nSpike>0:
        spikeDF['BPD']=wellDF['BPD'][sidx]
        spikeDF['Date']=wellDF['Date'][sidx]
        spikeDF['InjectionWellId']=wellDF['InjectionWellId'][sidx]
        spikeDF['ID']=wellDF['ID'][sidx]
        spikeDF['Modeled']=wellDF['Modeled'][sidx]
      if nKeep>0:
        keepDF['BPD']=wellDF['BPD'][kidx]
        keepDF['Date']=wellDF['Date'][kidx]
        keepDF['InjectionWellId']=keepDF['InjectionWellId'][kidx]
        keepDF['ID']=wellDF['ID'][kidx]
        keepDF['Modeled']=wellDF['Modeled'][kidx]
      if nMod>0:
        modDF['BPD']=wellDF['BPD'][midx]
        modDF['Date']=wellDF['Date'][midx]
        modDF['InjectionWellId']=wellDF['InjectionWellId'][midx]
        modDF['ID']=wellDF['ID'][midx]
        modDF['Modeled']=wellDF['Modeled'][midx]
  ################################
  # Return separated data frames #
  ################################
  return(keepDF,spikeDF,nanDF,modDF)

def getDays(inDF,epoch):
  ############################################################
  # Creates a column of integer day numbers from input dates #
  ############################################################
  # Inputs                                      #
  #    inDF - Dataframe with Date column        #
  #    epoch - datetime value of starting date  #
  # Outputs                                     #
  #    outDF - Dataframe with added Days column #
  ###############################################
  outDF=inDF.copy()
  dayList=[]
  ##################
  # Loop over rows #
  ##################
  for index,row in inDF.iterrows():
    #######################
    # Access Dates column #
    #######################
    date=pd.to_datetime(row['Date'])
    try:
      dayList.append((date-epoch).days)
    except:
      print('getDays error ',date,epoch)
  outDF['Days']=dayList
  return outDF

def regularize(wellDF,endDay,interval=1,verbose=0):
  #########################################
  # Time regularization of injection data #
  #########################################
  # Check if dataframe is empty #
  ###############################
  if wellDF.shape[0]==0: print("wellDF has no injection")
  ########################
  # Initialize output DF #
  ########################
  interpDF=pd.DataFrame(columns=['ID','Days','BPD'])
  ########################
  # Make lists of values #
  ########################
  days=wellDF['Days'].to_list()
  bpds=wellDF['BPD'].to_list()
  ids=wellDF['ID'].to_list()
  lastBPD=bpds[-1]
  ############################
  # Find starting output day #
  ############################
  minDay=np.nanmin(days)-interval
  ###############################
  # This will be a back-average #
  ###############################
  outStartDay=interval*(1+math.floor(minDay/interval))
  outEndDay=interval*math.ceil(endDay/interval)
  ###################
  # Get output days #
  ###################
  nDaysOut=1+round((outEndDay-outStartDay)/interval)
  outDays=np.linspace(outStartDay,outEndDay,num=nDaysOut)
  outBBL=np.zeros([nDaysOut,1])
  if len(days)==1:
    ################################################
    # If one entry, assume the same BPD for output #
    # Not sure if this is a good assumption or not #
    ################################################
    outBBL=0*outDays+bpds[0]
    if verbose>0: print(' injectionV3.regularize: one value for well')
  elif len(days)==0:
    if verbose>0: print(' injectionV3.regularize: empty output')
    outBBL=0*outDays
  else:
    #########################################
    # First generate a daily injection rate #
    #########################################
    daily=np.linspace(minDay,endDay,num=(endDay-minDay+1))
    interpBBL=interpolate.interp1d(days,bpds,kind='next',fill_value=(0.,lastBPD),bounds_error=False)
    dailyBBL=interpBBL(daily)
    if verbose>1: print(' injectionV3.regularize: Daily result from ',minDay,' to ',endDay,'; output from ',outStartDay,' to ',outEndDay,' at ',interval)
    #############################################
    # Now sum the rate over the output interval #
    #############################################
    for id in range(nDaysOut):
      ################################
      # Get index of starting date   #
      # This is a backward average   #
      # Truncate for the first value #
      ################################
      startIndex=outStartDay-minDay+(id-1)*interval
      endIndex=startIndex+interval
      startIndex=max(startIndex,0)
      outBBL[id]=np.sum(dailyBBL[startIndex:endIndex])/float(interval)
    outBBL=outBBL.flatten()
  ##############################
  # Copy result into dataframe #
  ##############################
  interpDF['BPD']=outBBL
  interpDF['Days']=outDays
  interpDF['ID']=ids[0]
  return interpDF

def writeDictDF(dictDF,OutFile,verbose=0):
  #######################################################
  # Output a dictionary where each entry is a dataframe #
  #######################################################
  if verbose>0: print (' injectionV3.writeDictDF: ',OutFile)
  f=0
  for well in dictDF:
    if dictDF[well].shape[0]>0:
      if f==0:
        ####################################
        # For first entry, create the file #
        ####################################
        dictDF[well].to_csv(OutFile,index=False)
      else:
        #################################
        # For subequent entries, append #
        #################################
        dictDF[well].to_csv(OutFile,index=False,mode='a',header=False)
      f=f+1
  return