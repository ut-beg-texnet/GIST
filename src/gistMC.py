# Written for Python 3.6
# Copyright 2022 ExxonMobil Upstream Research Company
# Authors: Prototype - Lei Jin; Port - Bill Curry
# Package Depenencies - pandas, scipy, numpy

# Well function for pore pressure solution
import scipy.special as sc

# Interpolator for volumes time series for poroelastic solution
import scipy.interpolate as si

# Base libraries
import numpy as np
import pandas as pd
import math

# List manipulation - probably not needed
from itertools import compress

class gistMC:
  
  def __init__(self,minZ,maxZ,epoch=pd.to_datetime('01-01-1980'),nReal=100,seed=42):
    # Initialize base class
    # Set internal parameters that shouldn't need to be changed
    self.g=9.81         # acceleration due to gravity m/s**2
        
    self.ntBin=51       # Number of bins to interpolate for each injection interval for numerical integration for poroelastic case
    # Well depth filtering parameters
    # Future work: Make this pull horizons 
    # and lookup for each well instead of 
    # constant values 
    self.minZ=minZ      # Minimum depth of interval in feet from MSL
    self.maxZ=maxZ      # Maximum depth of interval in feet from MSL
    
    #Time parameters
    self.epoch=epoch # Reference time/date for injection data
    
    # Number of realizations
    self.nReal=nReal
    
    # Initialize random number generator
    self.rng=np.random.default_rng(seed=seed)
    
    # Initialize well data frame
    self.nw=0
    # This will need to change when we move the well information out-of-core
    self.wells=pd.DataFrame(columns=['API','Name','Operator','FirstDate','LastDate','TotalBBL','InjectionDays','AvgBPD','TopDepth','BottomDepth','DiffusionDist','Latitude','Longitude','Dates','Days','BPDs'])
    
    # Initialize impulse response arrays - to do
    # Rate, duration, sampling time
    # Radial sampling,
    # orientation of fault
    # To-do: Error checking
  
  def initPP(self,rho_0=1000.,nta=1.e-3,phi=10.,kMD=200.,h=100.,alphav=1.08e-9,beta=3.6e-10):
    # Set baseline pore pressure parameters
    self.phi=phi        # Porosity
    self.kMD=kMD        # Permeability in millidarcies
    self.rho=rho_0      # Density of fluid in kg/m**3
    self.nta=nta        # Fluid viscosity  in Pa . s
    self.alphav=alphav  # Vertical compressibility of aquifer in 1/Pa
    self.beta=beta      # Fluid compressibility 1/Pa
    self.h=h            # Interval thickness in ft
    # Unit conversions, intermediate parameters:
    (phiFrac,hM,kapM2,S,K,T,diffPP)=calcPPVals(kMD,h,alphav,beta,phi,rho_0,self.g,nta)
    self.phiFrac=phiFrac
    self.hM=hM
    self.kapM2=kapM2
    self.S=S
    self.T=T
    self.diffPP=diffPP
    # To-do - error checking - bounds of parameters, unphysical rock, etc.
  
  def initPE(self,mu=20e9,nu=0.25,nu_u=0.3,alpha=0.31,mu_f=0.6,mu_r=0.6,match=False):
    # Poroelastic parameters
    self.mu=mu          # Rock shear modulus - same as G
    self.nu=nu          # Rock drained Poisson's ratio - unitless
    self.nu_u=nu_u      # Rock undrained Poisson's ratio - unitless
    self.alpha= alpha   # Biot's coefficient - unitless
    self.muF=mu_f      # Fault friction coefficient - unitless
    self.muR=mu_r      # Rock friction coefficient - unitless
    self.match=match
    # Compute
    if match:
      pass
      # Compute based on matching impulse responses
      #    scalar=matchPE2PP
      #    Recompute PE params - lamda_uVec, then diffPE, nu_u, and B
    # To-do - error checking - bounds of parameters, unphysical rock, etc.
    
  def initMCPP(self,rhoUnc=2.,ntaUnc=1e-6,phiUnc=3.,kMDUnc=50.,hUnc=25.,alphavUnc=1e-11,betaUnc=1e-12):
    # Set internal parameters
    self.rhoUnc=rhoUnc
    self.ntaUnc=ntaUnc
    self.phiUnc=phiUnc
    self.kMDUnc=kMDUnc
    self.hUnc=hUnc
    self.alphavUnc=alphavUnc
    self.betaUnc=betaUnc
    # To-do - error checking on uncertainty numbers - make sure it's physical
    # To-do - add more PDFs vs. flat
    # Generate vector of nReal x 13 random floats - note that upper bound is outside of range
    self.randomFloats=self.rng.random(size=(self.nReal,13))
    self.rhoVec   =self.randomFloats[:,0]*rhoUnc*2.   +(self.rho   -self.rhoUnc)
    self.ntaVec   =self.randomFloats[:,1]*ntaUnc*2.   +(self.nta   -self.ntaUnc)
    self.phiVec   =self.randomFloats[:,2]*phiUnc*2.   +(self.phi   -self.phiUnc)
    self.kMDVec   =self.randomFloats[:,3]*kMDUnc*2.   +(self.kMD   -self.kMDUnc)
    self.hVec     =self.randomFloats[:,4]*hUnc*2.     +(self.h     -self.hUnc)
    self.alphavVec=self.randomFloats[:,5]*alphavUnc*2.+(self.alphav-self.alphavUnc)
    self.betaVec  =self.randomFloats[:,6]*betaUnc*2.  +(self.beta  -self.betaUnc)
    (phiFracVec,hMVec,kapM2Vec,SVec,KVec,TVec,diffPPVec)=calcPPVals(self.kMDVec,self.hVec,self.alphavVec,self.betaVec,self.phiVec,self.rhoVec,self.g,self.ntaVec)
    self.phiFracVec=phiFracVec
    self.hMVec=hMVec
    self.kapM2Vec=kapM2Vec
    self.SVec=SVec
    self.KVec=KVec
    self.TVec=TVec
    self.diffPPVec=diffPPVec
    self.diffPPMax=np.max(diffPPVec)
    # To-do - should output realizations of parameters here - make data frame
    # Error checking
    print(" Monte Carlo pore pressure - rho    min/max:",np.amin(self.rhoVec),np.amax(self.rhoVec))
    print(" Monte Carlo pore pressure - nta    min/max:",np.amin(self.ntaVec),np.amax(self.ntaVec))
    print(" Monte Carlo pore pressure - phi    min/max:",np.amin(self.phiVec),np.amax(self.phiVec))
    print(" Monte Carlo pore pressure - kMD    min/max:",np.amin(self.kMDVec),np.amax(self.kMDVec))
    print(" Monte Carlo pore pressure - h      min/max:",np.amin(self.hVec),np.amax(self.hVec))
    print(" Monte Carlo pore pressure - alphav min/max:",np.amin(self.alphavVec),np.amax(self.alphavVec))
    print(" Monte Carlo pore pressure - beta   min/max:",np.amin(self.betaVec),np.amax(self.betaVec))
    print(" Monte Carlo pore pressure - kapM2  min/max:",np.amin(self.kapM2Vec),np.amax(self.kapM2Vec))
    print(" Monte Carlo pore pressure - S      min/max:",np.amin(self.SVec),np.amax(self.SVec))
    print(" Monte Carlo pore pressure - K      min/max:",np.amin(self.KVec),np.amax(self.KVec))
    print(" Monte Carlo pore pressure - T      min/max:",np.amin(self.TVec),np.amax(self.TVec))
    print(" Monte Carlo pore pressure - diffPP min/max:",np.amin(self.diffPPVec),np.amax(self.diffPPVec))
    return
    
  def initMCPE(self,muUnc=1e9,nuUnc=0.02,nu_uUnc=0.02,alphaUnc=0.05,muFUnc=0.05,muRUnc=0.05,match=False):
    self.muUnc=muUnc
    self.nuUnc=nuUnc
    self.nu_uUnc=nu_uUnc
    self.alphaUnc=alphaUnc
    self.muFUnc=muFUnc
    self.muRUnc=muRUnc
    self.muVec    =self.randomFloats[:,7]*muUnc*2.   +(self.mu   -self.muUnc)
    self.nuVec    =self.randomFloats[:,8]*nuUnc*2.   +(self.nu   -self.nuUnc)
    self.nu_uVec  =self.randomFloats[:,9]*nu_uUnc*2.   +(self.nu_u -self.nu_uUnc)
    self.alphaVec =self.randomFloats[:,10]*alphaUnc*2.  +(self.alpha-self.alphaUnc)
    self.muFVec   =self.randomFloats[:,11]*muFUnc*2.  +(self.muF  -self.muFUnc)
    self.muRVec   =self.randomFloats[:,12]*muRUnc*2.  +(self.muR  -self.muRUnc)
    (lamdaVec,lamda_uVec,BVec,diffPEVec,kappaVec)=calcPEVals(self.muVec,self.nuVec,self.nu_uVec,self.alphaVec,self.kapM2Vec,self.ntaVec)
    self.lamdaVec=lamdaVec
    self.lamda_uVec=lamda_uVec
    self.BVec=BVec
    self.diffPEVec=diffPEVec
    self.kappaVec=kappaVec
    print(" Monte Carlo poroelastic - mu    min/max:",np.amin(self.muVec),np.amax(self.muVec))
    print(" Monte Carlo poroelastic - nu    min/max:",np.amin(self.nuVec),np.amax(self.nuVec))
    print(" Monte Carlo poroelastic - nu_u  min/max:",np.amin(self.nu_uVec),np.amax(self.nu_uVec))
    print(" Monte Carlo poroelastic - alpha min/max:",np.amin(self.alphaVec),np.amax(self.alphaVec))
    print(" Monte Carlo poroelastic - muF   min/max:",np.amin(self.muFVec),np.amax(self.muFVec))
    print(" Monte Carlo poroelastic - muR   min/max:",np.amin(self.muRVec),np.amax(self.muRVec))
    print(" Monte Carlo poroelastic - lamda min/max:",np.amin(self.lamdaVec),np.amax(self.lamdaVec))
    print(" Monte Carlo poroelastic -lamda_umin/max:",np.amin(self.lamda_uVec),np.amax(self.lamda_uVec))
    print(" Monte Carlo poroelastic - B     min/max:",np.amin(self.BVec),np.amax(self.BVec))
    print(" Monte Carlo poroelastic -diffPE min/max:",np.amin(self.diffPEVec),np.amax(self.diffPEVec))
    print(" Monte Carlo poroelastic - kappa    min/max:",np.amin(self.kappaVec),np.amax(self.kappaVec))
    if match:
      pass
      # To-do - Loop over values and find impulse response
      # for ir in range(self.nReal):
      #    scalar=matchPE2PP
      #    Recompute PE params - lamda_uVec, then diffPE, nu_u, and B
    # To-do - should output realizations of parameters here - make data frame
    return
  
  def writeRealizations(self,path):
    # Make data frame of different parameter realizations for pore pressure
    # " for poroelastic case
    realization=np.arange(self.nReal)
    d={'rho':self.rhoVec,'nta':self.ntaVec,'phi':self.phiVec,'kMD':self.kMDVec,'h':self.hVec,'alphav':self.alphavVec,'beta':self.betaVec,'kapM2':self.kapM2Vec,'S':self.SVec,'T':self.TVec,'K':self.KVec,'diffPP':self.diffPPVec,'mu':self.muVec,'nu':self.nuVec,'nu_u':self.nu_uVec,'alpha':self.alphaVec,'muF':self.muFVec,'muR':self.muRVec,'lambda':self.lamdaVec,'lambdaU':self.lamda_uVec,'B':self.BVec,'diffPE':self.diffPEVec,'kappa':self.kappaVec,'realization':realization}
    outDataFrame=pd.DataFrame(data=d)
    outDataFrame.to_csv(path)
    return
  
  def matchPE2PP(self,a=0.01,b=100.,tol=0.01,niter=500):
    # Compute known pore pressure impulse response
    ipp=impResPP()
    # Find scalar to match impulse responses via bisection
    for iter in range(0,niter):
      c=(a+b)/2.
      fc=sum(impulseResponsePP(known)-impulseResponsePoro(c))
      fa=sum(impulseResponsePP(known)-impulseResponsePoro(a))
      if fc==0. or (b-a)/2.<tol: return c
      if np.sign(fc)==np.sign(fa):
        a=c
      else:
        b=c
    return c

  def addTXWells(self,txWells,prefix=''):  
    # TO BE DEPRECEATED WITH V3 DATA
    # Takes output of injection preprocessing code that regularizes injection reporting #
    # Based on v2 schema of B3 injection data, which has now changed #
    # To-do: switch to v3 of B3 injection data.
    # First pass of filtering:
    #   Well Type is Injection Into Non-Productive Zone
    #   Depths are within the interval
    wf=txWells[(txWells[prefix+'Type']=="Injection Into Non-Productive Zone") & (txWells[prefix+'TopDepth']>self.minZ) & (txWells[prefix+'BottomDepth']<self.maxZ)].reset_index(drop=True)
    # Create new dataframe to append to self.wells
    tdf=pd.DataFrame(columns=['API','Name','Operator','FirstDate','LastDate','TotalBBL','InjectionDays','AvgBPD','TopDepth','BottomDepth','DiffusionDist','Latitude','Longitude','Dates','Days','BPDs'])
    # Get unique list of wells here - then create a new DF from that
    # Get unique API values - this isn't ideal
    # To-do: use the unique B3 identifier as API has a group of wells listed as well 0 within a county 
    APIs=wf['API'].unique()
    nAPI=len(APIs)
    # Loop over each API
    nSkip=0
    for api in APIs:
      # Get subset of dataframe for wells only with one API
      apidf=wf[(wf['API']==api)].reset_index(drop=True)
      # To-do: check to see that the other values are consistent - UIC, Num, Name, Operator, Status, etc.
      # Pull all injection dates days and rates for that API = each of these should be vectors/lists
      wDict={} # Dictionary for this individual well
      wDict['API']=api
      wDict['Days']=apidf['Days']
      wDict['BPDs']=apidf['DailyBBL']
      wDict['Dates']=apidf['Dates']
      # Take time series and compute: 'FirstDate','LastDate','TotalBBL','InjectionDays','AvgBPD','DiffusionDist'
      # This should be a function!
      # Get time step
      # !!! Note that we assume these data are regularized and have a consistent time step !!!
      dT=apidf['Days'][1]-apidf['Days'][0]
      # Get number of nonzero rates for this well
      nBPDs=np.count_nonzero(wDict['BPDs'].to_numpy())
      # If all rates are zero, we can skip this well
      if nBPDs==0:
        #print("skipping well #",api)
        nSkip=nSkip+1
        continue
      # Get first and last injection rates
      firstI=wDict['BPDs'].index[wDict['BPDs'].to_numpy().nonzero()[0][0]]
      lastI=wDict['BPDs'].index[wDict['BPDs'].to_numpy().nonzero()[0][-1]]
      wDict['FirstDate']=wDict['Dates'][firstI]
      wDict['LastDate']=wDict['Dates'][lastI]
      # Get an overall time interval between first and last injection dates
      # Note - this isn't quite what we want - we want this distance to be relative to the EQ time
      wDict['InjectionDays']=wDict['Days'][lastI]-wDict['Days'][firstI]
      wDict['TotalBBL']=dT*wDict['BPDs'].sum()
      wDict['AvgBPD']=wDict['TotalBBL']/wDict['InjectionDays']
      # Note - I should probably use TOday instead of last date, but wanted reproducible results
      #wDict['DiffusionDist']=0.001*np.sqrt(4.*np.pi*self.diffusivity*wDict['InjectionDays']*24*60*60)
      # Change diffusivity to diffPPMax - might include poroelastic case although it's different
      wDict['DiffusionDist']=0.001*np.sqrt(4.*np.pi*self.diffPPMax*wDict['InjectionDays']*24*60*60)
      # Get other items: 'Name','Operator','Latitude','Longitude'
      wDict['Name']=apidf['Name'][0]
      wDict['Operator']=apidf['Operator'][0]
      wDict['Latitude']=apidf['Latitude'][0]
      wDict['Longitude']=apidf['Longitude'][0]
      # Append
      self.wells=self.wells.append(wDict,ignore_index=True)
    self.nw=self.nw+nAPI-nSkip
    # Post some information - number of wells added, diffusion distances, kbd ranges, etc.
    print(nAPI," wells considered - total ",self.nw,", skipped ",nSkip)
    print(' Wells columns ',self.wells.columns)
    return

  def checkWells(self,name):
    # TO BE DEPRECEATED WITH V3 DATA
    # Post number of non-zero and zero volume wells
    self.checkWellValue(name,'TotalBBL')
    # Post number of invalid FirstDate and LastDate values
    self.checkWellDate(name,'FirstDate')
    self.checkWellDate(name,'LastDate')
    # Post number of invalid DiffusionDist values
    self.checkWellValue(name,'DiffusionDist')
    # Post number of invalid BPD values
    self.checkWellValue(name,'BPD')
    # Post coordinates
    self.checkWellValue(name,'Latitude')
    self.checkWellValue(name,'Longitude')

  def checkWellDate(self,name,column):
    # TO BE DEPRECEATED WITH V3 DATA
    print(name," ",column," Length:",self.wells.shape[0]," Oldest:",self.wells[column].min()," Newest: ",self.wells[column].max()," NaNs: ",self.wells[column].isna().sum())
      
  def checkWellValue(self,name):
    # TO BE DEPRECEATED WITH V3 DATA
    print(name," ",column," Length:",self.wells.shape[0]," Zeros:",self.wells.shape[0]-self.wells[column].astype(bool).sum(axis=0)," Max:",self.wells[column].max()," NaNs: ",self.wells[column].isna().sum())

  def findWells(self,eq):
    ###########################################################
    # Get subset of wells within contribution distance/time   #
    ###########################################################
    # eq: input earthquake dictionary - assumed columns from TexNet csv output
    # To-do: remove well selection and make separate subroutine
    
    # Step 1: Get wells that might have contributed
    # Find distances from eq to all wells
    distances=np.zeros([self.nw])
    dxs=np.zeros([self.nw])
    dys=np.zeros([self.nw])
    # Get uncertainty for the earthquake - note that this is one standard deviation so we are not being as conservative as we should be
    # To-do - could incorporate azimuth to better account for the azimuthal variation of uncertainty - sandbag in the meantime
    eqUncert=math.sqrt(eq['Latitude Error (km)']**2+eq['Longitude Error (km)']**2)
    for iw in range(self.nw):
      # Find distance from well to earthquake
      distances[iw]=haversine(eq['Latitude (WGS84)'],self.wells['Latitude'][iw],eq['Longitude (WGS84)'],self.wells['Longitude'][iw])
      [dxs[iw],dys[iw]]=haversineXY(eq['Latitude (WGS84)'],self.wells['Latitude'][iw],eq['Longitude (WGS84)'],self.wells['Longitude'][iw])

    # Only include wells that have the earthquake inside of the characteristic distance - can include event uncertainty here
    consideredWells=self.wells[self.wells['DiffusionDist']>(distances-eqUncert)].reset_index(drop=True)
    consideredWells['Distances']=distances[self.wells['DiffusionDist']>(distances-eqUncert)]
    consideredWells['DXs']=dxs[self.wells['DiffusionDist']>(distances-eqUncert)]
    consideredWells['DYs']=dys[self.wells['DiffusionDist']>(distances-eqUncert)]
    return consideredWells
    
  def runPressureScenarios(self,eq,consideredWells,prefix=''):
    # Take earthquake origin time and convert it to days since epoch
    eqDay=(eq['Origin Date']-self.epoch).days
    
    # Known issue: we do not compute diffusion distance relative to the time of event!
    # We currently compute it relative to the last injection time, which isn't ideal
    
    # Post number of wells considered for this earthquake
    nwC=consideredWells.shape[0]
    print("Number of wells considered: ",nwC)
    
    # Initialize MC pressure arrays
    pressures=np.zeros([nwC,self.nReal])
    percentages=np.zeros([nwC,self.nReal])
    totalPressures=np.zeros([nwC,self.nReal])
    
    # Form output dataframe of realizations
    #scenarios=pd.DataFrame(columns=['EventID','EventLatitude','EventLongitude','API','Name','Operator','Latitude','Longitude','NumWells','Pressures','TotalPressure','Percentages','Realization'])
    
    # Loop over wells in consideration
    for iwc in range(nwC):
      # compute pressures for each well
      bpds=consideredWells['BPDs'][iwc]
      # days are realtive to epoch
      days=consideredWells['Days'][iwc]
      # Convert distance from km to m
      dist=1000.*consideredWells['Distances'][iwc]
      # Pfrontvar is a reimplementation of the FSP pore pressure modeling code
      for iReal in range(self.nReal):
        pressures[iwc,iReal]=self.pressureScenario(bpds,days,eqDay,dist,iReal)
    # Form output dataframe of realizations
    scenarios=pd.DataFrame(columns=['EventID','EventLatitude','EventLongitude','API','Name','Operator','Latitude','Longitude','NumWells','Pressures','TotalPressure','Percentages','Realization'])
    # Loop over realizations
    for iReal in range(self.nReal):
      scenarioDF=pd.DataFrame(columns=['EventID','EventLatitude','EventLongitude','API','Name','Operator','Latitude','Longitude','NumWells','Pressures','TotalPressure','Percentages','Realization'])
      # Get total pressure for this realization
      totalPressures[:,iReal]=np.sum(pressures[:,iReal])
      # Compute percentages
      percentages[:,iReal]=100.*pressures[:,iReal]/totalPressures[:,iReal]
      nwCnz=sum(percentages[:,iReal]>1.)
      # Make a dataframe 
      scenarioDF['Pressures']=pressures[:,iReal]
      scenarioDF['TotalPressure']=totalPressures[:,iReal]
      scenarioDF['Percentages']=percentages[:,iReal]
      scenarioDF['EventID']=eq['EventID']
      scenarioDF['EventLatitude']=eq['Latitude (WGS84)']
      scenarioDF['EventLongitude']=eq['Longitude (WGS84)']
      scenarioDF['API']=consideredWells['API']
      scenarioDF['Name']=consideredWells['Name']
      scenarioDF['Operator']=consideredWells['Operator']
      scenarioDF['Latitude']=consideredWells['Latitude']
      scenarioDF['Longitude']=consideredWells['Longitude']
      scenarioDF['NumWells']=nwCnz
      # Add realization number
      scenarioDF['Realization']=iReal
      # Concatenate to scenarios
      scenarios=pd.concat([scenarios,scenarioDF],ignore_index=True)
    # Return scenarios
    return scenarios

  def pressureScenario(self,bpds,days,eqDay,r,iReal):
    nd=np.count_nonzero(days<eqDay)
    # If the earthquake was before any injection, return 0
    if nd==0: return 0.
    # Make arrays of time in seconds to earthquake, volume in m3/sec
    priorSec=24*60*60*(eqDay-days[days<eqDay]) # Seconds from injection time to earthquake day
    priorQ=1.84013e-6 * bpds[days<eqDay] #;% 1 oil barrel per day = 1.84013e-6 cubic meters/second according to google
    timeSteps=np.zeros([nd,])
    # First time step
    timeSteps[0]=sc.exp1((r*r*self.TVec[iReal]*self.SVec[iReal])/(4.*self.TVec[iReal]*self.TVec[iReal]*priorSec[0]))*priorQ[0]
    for i in range(1,nd):
      # Generate well function for this time interval - injection until earthquake
      pp = (r*r*self.TVec[iReal]*self.SVec[iReal])/(4.*self.TVec[iReal]*self.TVec[iReal]*priorSec[i]) # This takes the previous time and spreads it to the end
      wellFunc = sc.exp1(pp)
      # This takes the differences in the injection rate and spreads it across to the max time
      timeSteps[i] = wellFunc*(priorQ[i]-priorQ[i-1])
    # Sum all time steps
    head = sum(timeSteps)*(1./(4.*np.pi*self.TVec[iReal]))
    dP = head*self.rhoVec[iReal]*self.g
    # Convert to PSI
    dP=dP/6894.76
    # Check for negative pressure - I had an off-by-one bug earlier I caught this way
    if dP<0.: print("Negative pressure! ",dP,",",timeSteps)
    return dP

  def runPoroelasticScenarios(self,eq,consideredWells,prefix=''):
    #########################################################
    # Calculate relative contributions for poroelastic case #
    #########################################################
    # eq: input earthquake dictionary - assumed columns from TexNet csv output
    # To-do: remove well selection and make separate subroutine
    
    # Take earthquake origin time and convert it to days since epoch
    eqDay=(eq['Origin Date']-self.epoch).days
    # See comments in pore pressure case - we are using the last injection date of the well instead of the event date
    # I also need to compute time of first injection to time of event
    nwC=consideredWells.shape[0]
    print("Number of wells considered: ",nwC)
    
    # Initialize pressures and stresses - need 4 values for stress
    pressures=np.zeros([nwC,self.nReal])
    stresses=np.zeros([nwC,4,self.nReal])
    totalPressure=np.zeros([nwC,self.nReal])
    totalStresses=np.zeros([nwC,self.nReal])
    poroStress=np.zeros([nwC,self.nReal])
    percentages=np.zeros([nwC,self.nReal])
    # If we don't have a strike or dip of event
    # To-do: Pull from a grid of stress data what the most likely earthquake is at a location
    #        Need to worry about nonuniqueness of stress and fault plane information
    if np.isnan(eq['Strike']):
      az=90.
    else:
      az=eq['Strike']
    if np.isnan(eq['Dip']):
      dip=90.
    else:
      dip=eq['Dip']
    for iwc in range(nwC):
      # compute stresses for each well - don't use an index, use iloc
      bpds=consideredWells['BPDs'][iwc]
      days=consideredWells['Days'][iwc]
      # Move from km to m for distances
      dist=1000.*consideredWells['Distances'][iwc]
      dx=1000.*consideredWells['DXs'][iwc]
      dy=1000.*consideredWells['DYs'][iwc]
      for iReal in range(self.nReal):
        (pressures[iwc,iReal],stresses[iwc,:,iReal])=self.poroelasticScenario(bpds,days,eqDay,dx,dy,dist,az,dip,self.ntBin,False,iReal)
    # Form output dataframe of realizations
    scenarios=pd.DataFrame(columns=['EventID','EventLatitude','EventLongitude','API','Name','Operator','Latitude','Longitude','NumWells','Stresses','TotalStress','Percentages','Realization'])
    # Get relative contributions of poroelastic stress
    # Now loop over realizations
    for iReal in range(self.nReal):
      scenarioPE=pd.DataFrame(columns=['EventID','EventLatitude','EventLongitude','API','Name','Operator','Latitude','Longitude','NumWells','Stresses','TotalStress','Percentages','Realization'])
      (percentages[:,iReal],poroStress[:,iReal],totalStresses[:,iReal])=self.poroAttr(pressures[:,iReal],stresses[:,:,iReal],az,dip,iReal)
      # Attribution part - need to go from stresses and pressures to 
      print("runPoroelasticScenarios: ",iReal," poroStress: ",np.amax(poroStress[:,iReal])," poroPerc: ",np.amax(percentages[:,iReal])," TotalStress: ",np.amax(totalStresses[:,iReal]))
      scenarioPE['Stresses']=poroStress[:,iReal]
      scenarioPE['Percentages']=percentages[:,iReal]
      scenarioPE['TotalStress']=totalStresses[:,iReal]
      nwCnz=sum(scenarioPE['Percentages']>1.)
      scenarioPE['NumWells']=nwCnz
      scenarioPE['EventID']=eq['EventID']
      scenarioPE['EventLatitude']=eq['Latitude (WGS84)']
      scenarioPE['EventLongitude']=eq['Longitude (WGS84)']
      scenarioPE['API']=consideredWells['API']
      scenarioPE['Name']=consideredWells['Name']
      scenarioPE['Operator']=consideredWells['Operator']
      scenarioPE['Latitude']=consideredWells['Latitude']
      scenarioPE['Longitude']=consideredWells['Longitude']
      # Add realization number
      scenarioPE['Realization']=iReal
      # Concatenate to scenarios
      scenarios=pd.concat([scenarios,scenarioPE],ignore_index=True)
    # Return scenarios
    return scenarios
  
  def poroelasticScenario(self,bpds,days,eqDay,dx,dy,r,azimuth,dip,ntBin,recomp,iReal):
    ########################
    # Poroelastic modeling #
    ########################
    #Inputs
    # bpd - np array or list of barrels per day
    # days - np array or list of days of bpd numbers relative to self.epoch
    ## bpd and days must be regularly sampled in time!
    # eqday - integer of day of earthquake relative to self.epoch
    # dx, dy - floats of x and y distance from well to earthquake
    # r - float of distance from well to earthquake - I can remove this if I have dx and dy
    # azimuth of fault plane (strike) in degrees 0-360 - 0 = North, clockwise
    # dip of fault plane - 0-90 - I think right-hand convention (tested on vertical faults mostly so far
    # ntBin - how densely to resample the injection time series for numerican integration
    # recomp - do we use the original or recomputed undrained Lame's constant?
    # iReal - realization number
    ################################
    # Outputs
    # dP - change in pore pressure
    # dS - change in stress - [4,1]
    ############################
    dP=0.; dS=0.
    #Get number of time steps
    nd=np.count_nonzero(days<eqDay)
    # If no injection before earthquake, return zeroes
    if nd==0: return (0.,[0.,0.,0.,0.])
    # Convert bpds and days to m3/s/m and s
    sec=24*60*60*(days[days<eqDay]) # Seconds from injection time relative to epoch
    eqSec=eqDay*24*60*60
    dt=sec[1]-sec[0]
    dtp=dt/ntBin
    
    #q=self.rou_0*1.84013e-6 * bpds[days<eqDay] #;% 1 oil barrel per day = 1.84013e-6 cubic meters/second according to google
    q=self.rhoVec[iReal]*1.84013e-6 * bpds[days<eqDay] #;% 1 oil barrel per day = 1.84013e-6 cubic meters/second according to google
    # q here should be in kg/s/m? - yes - need to multiply by density in kg/m3 to get kg/s
    # Go from mass/time to mass/time/length
    qp=q/self.hVec[iReal] # go from kg/s to kg/s/m
    # Make interpolated time series - inputs are the original data
    qinterp=si.interp1d(sec,qp,kind='previous',fill_value="extrapolate")
    # tp is the desired output time series
    tp=np.arange(0.,eqSec,dtp)
    # call interpolation function
    qptp=qinterp(tp)
    # Compute constants: I should compute these in the initialize phase
    C1= 1./(4.*np.pi*self.rhoVec[iReal]*self.kappaVec[iReal]); 
    # do we used the recomputed or original undrained Lame's constant?
    # Not implemented for realizationsVec[iReal]
    C2= self.muVec[iReal]*(self.lamda_uVec[iReal]-self.lamdaVec[iReal])/(np.pi*self.alphaVec[iReal]*self.rhoVec[iReal]*(self.lamda_uVec[iReal]+2*self.muVec[iReal])*r**2)
    
    #if recomp:
    #  C2= self.mu*(self.l_u_new-self.lamda)/(np.pi*self.alpha*self.rou_0*(self.l_u_new+2*self.mu)*r**2) # rou_0 was missing in Rudnicki (1986) Eq.(45)
    #else:
    #  C2= self.mu*(self.lamda_u-self.lamda)/(np.pi*self.alpha*self.rou_0*(self.lamda_u+2*self.mu)*r**2) # rou_0 was missing in Rudnicki (1986) Eq.(45)
    # Initialize pressure and stresses
    pw=np.zeros([nd,])
    Sw_xx=np.zeros([nd,])
    Sw_yy=np.zeros([nd,])
    Sw_xy=np.zeros([nd,])
    Sw_xz=np.zeros([nd,])
    Sw_yz=np.zeros([nd,])
    # Option for recomputed or original diffusivity
    #if recomp:
    #  # tp and koxi are both vectors here
    #  koxi=r/np.sqrt(self.diffusivityPENew*(eqSec-tp)) 
    #else:
    #  koxi=r/np.sqrt(self.diffusivityPE*(eqSec-tp))
    # To-do - include recomputed values
    if (self.diffPEVec[iReal]<0): print("poroelasticScenario: diffPE negative!",self.diffPEVec[iReal])
    
    koxi=r/np.sqrt(self.diffPEVec[iReal]*(eqSec-tp))
    G1 =np.exp(-0.25*koxi*koxi)         #% vector 
    G2 =0.5*koxi*koxi                #% vector
    # Numerical integration over time up to this point 
    #change EQ.x(i), EQ.y(i), EQ.z(i) into Xv(i), Yv(i), Zv(i) if
    # calculating and recording at all meshgrid nodes 
    # qptp - interpolated rates, tp - time vector, G1 - vector
    F1=qptp/(eqSec-tp)*G1*dtp
    F2_xx=qptp*((1-2*dx*dx/(r*r))*(1-G1)-(1-(dx*dx)/(r*r))*G2*G1)*dtp 
    F2_yy=qptp*((1-2*dy*dy/(r*r))*(1-G1)-(1-(dy*dy)/(r*r))*G2*G1)*dtp
    F2_xy=qptp*((0-2*dx*dy/(r*r))*(1-G1)-(0-(dx*dy)/(r*r))*G2*G1)*dtp
    # Sum results for integration
    F1_int=np.sum(F1)
    F2_xx_int=np.sum(F2_xx)
    F2_yy_int=np.sum(F2_yy)
    F2_xy_int=np.sum(F2_xy)  
    
    #Record results by each well 
    # Pore pressure results for this well
    pw= C1*F1_int
    # Convert fro Pa to PSI
    dP=pw/6894.76 
    # Total stresses
    Sw_xx=C2*F2_xx_int *(-1)               #% xx, x-normal; convert to compression postive 
    Sw_yy=C2*F2_yy_int *(-1)               #% yy, y-normal; convert to compression postive 
    Sw_xy=C2*F2_xy_int                     #% xy, shear
    #  xz, shear, =0 under plain strain
    #  yz, shear, =0 under plain strain
    Sw_zz=self.nu*(Sw_xx+Sw_yy)
    dS=[Sw_xx/6894.76,Sw_yy/6894.76,Sw_xy/6894.76,Sw_zz/6894.76] # Pa to PSI):
    return(dP, dS)
    
  def poroAttr(self,pressures,stresses,azimuth,dip,iReal):
    ###########################
    # Poroelastic attribution #
    ###########################
    # Inputs
    # pressures - all pressures for all wells for one earthquake
    # stresses - all stresses "
    # azimuth - azimuth of focal plane, either from earthquake or from regional stress
    #         - convention of 0 degrees as N
    # dip     - dip of focal plane
    #         - convention of 90 degrees vertical, 0 flat
    # iReal   - realization number
    ###########
    # Outputs #
    # perc   - percentage contribution - [nwells,1]
    # CFF    -   
    # CFFsum - 
    ###########
    # First convert azimuth and dip to radians
    azr=np.radians(azimuth)
    dipr=np.radians(dip)
    
    # Get number of wells with contributing pressures/stresses
    nw=len(pressures)
    
    # Initialize CFF and percentages
    CFF=np.zeros([nw,1])
    perc=np.zeros([nw,1])
    CFFsum=np.zeros([nw,1])
    # Stresses and pressures are per-well - need to form total effective stress tensor
    ssv=np.sum(stresses,axis=0)*6894.76 # psi back to Pa
    pp=np.sum(pressures)* 6894.76 # psi back to Pa
    # Now form the simple effective stress tensor for this earthquake for all wells
    ss=formEffectiveStressTensor(ssv,pp)
    # Need to determine nf, tr1 and tr2 - phi=azimuth delta=dip
    nf=[np.sin(dipr)*np.cos(azr), -np.sin(dipr)*np.sin(azr), np.cos(dipr)]
    tf1=[np.sin(azr), np.cos(azr), 0.]
    tf2=[-np.cos(dipr)*np.cos(azr), np.cos(dipr)*np.sin(azr), np.sin(dipr)]
    # traction, normal & two shear stresses by all wells @ eth source fault on kth time step, Pa
    (tr,sn,st1,st2) = projectStress(ss,nf,tf1,tf2)
    # theta (see derivation), indicates final slip tendency direction
    theta=np.degrees(np.arctan(st2/st1))
    for iw in range(nw):
      # Form effective stress tensor for each well contributing to this earthquake
      ssw=formEffectiveStressTensor(stresses[iw,:],pressures[iw])
      # traction, normal & two shear stresses by nth wells @ eth source fault on kth time step, Pa  
      (trw,snw,st1w,st2w)=projectStress(ssw,nf,tf1,tf2)
      
      # theta by nth well (see derivation), indicates slip tendency direciton caused by this well 
      # if no shear stress (st1w=st2w=0), then return NaN
      theta_w=np.degrees(np.arctan(st2w/st1w))
      # beta by nth well, see derivation
      beta=theta_w-theta        # degrees 
               
      # fault-normal part of CFF by nth well, Pa 
      CFF_n= -self.muFVec[iReal]*snw
      
      # fault-tangential part of CFF by nth well, Pa 
      #CFF_t=sqrt(norm(trw,2)^2-snw^2);  # equivalent to sqrt(st1w^2+st2w^2)
      CFF_t=np.sqrt(np.linalg.norm(trw)**2-snw**2)
      # "relevant" fault-tangential part of CFF by nth well 
      # i.e.. along the final slip tendency direction, Pa 
      # only need to modify with bata when it is not nan
      if np.isnan(beta)==False:
          CFF_t=CFF_t*np.cos(np.radians(beta))
      # "relevant" CFF at eth source fault on kth time step by nth well, Pa 
      CFF[iw,0]=CFF_t+CFF_n
 
      # Contribution 
    #% total CFF by all wells at eth source faults on kth time step 
    CFFsum=np.sum(CFF)*np.ones([nw,1])
    for iw in range(nw):
      perc[iw,0]=100.*CFF[iw,0]/CFFsum[iw,0]  # EQ (Space) by Wells by Time 
    return (perc[:,0],CFF[:,0],CFFsum[:,0])

# Placeholders for supporting v3 injection data:
  #def addWells(self,wellFile,injFile):
    ## For now use previously-processed data - 
    ## Read wellFile
    ## Store well numbers, locations and first dates - 
    ## Compute diffusion distances given diffPPMax    
    #return
  
  #def findWells(self,eq,injFile):
    ## get well numbers from injection file
    #return
  
  #def getInjection(self,wells):
    ## get monthly and daily injection information from listed well numbers
    #return
  
  def impResPP(self):
    pass
  def impResPE(self):
    pass
  
    # 
  #addEQs
  #addWells - should be general and output from wellproc package
  
# Generic subroutines not inside the class

def calcPPVals(kMD,hFt,alphav,beta,phi,rho,g,nta):
  # Convert inputs to SI units and generate other parameters
  # Unit conversions
  kapM2 = kMD*(1e-3)*(9.9e-13)   # Convert from Millidarcies to m2
  hM=hFt*0.3048                       # ft to m
  phiFrac=phi*0.01
  # Now compute intermediate parameters
  diffPP=kapM2/(nta*(alphav+beta)*phiFrac) # Diffusivity for pore pressure case
  S=rho*g*hM*(alphav+phiFrac*beta)         # Storativity
  K = kapM2*rho*g/nta                      #saturated hydraulic conductivity
  T = K*hM                                 #Transmissivity
  return (phiFrac,hM,kapM2,S,K,T,diffPP)
  
def calcPEVals(mu,nu,nu_u,alpha,kapM2,nta):
  # Generate other parameters - PP vals already converted weird units to SI
  lamda=2.*mu*nu/(1.-2.*nu)                   # Rock drained Lame's constant, Pa; alternative parameter to nu
  lamda_u=2.*mu*nu_u/(1.-2.*nu_u)             # Rock undrained Lame's constant, Pa; alternative to nu_u
  B=3.*(nu_u-nu)/(alpha*(1.+nu_u)*(1.-2.*nu)) # Rock Skempton coefficientfor Berea sandstone, Hart & Wang [1995] report average 0.75
  diffPE=(kapM2)*(lamda_u-lamda)*(lamda+2.*mu)/(nta*alpha*alpha*(lamda_u+2.*mu)) # Diffusivity for Poroelastic case - m2 / s
  kappa=kapM2/nta                             # hydraulic conductivity, m^2/(pa*s) - I'm not sure that this gets used anywhere
  return (lamda,lamda_u,B,diffPE,kappa)

def haversineXY(lat1,lat2,lon1,lon2):
    # Assume that lat1 is the earthquake/investigation point and lat2 is the well location
    # Negative values assume the earthquake is South / West of the well (I think)
    # Great-circle distance between two lat/lon pairs - broken into x and y components for poroelastic modeling
    # Assumes sea level, probably some other spherical geometry issues, prolateness of earth, etc. #
    dy=haversine(lat1,lat2,0.5*(lon1+lon2),0.5*(lon1+lon2))
    dx=haversine(0.5*(lat1+lat2),0.5*(lat1+lat2),lon1,lon2)
    if lat1<lat2: dy=-dy
    if lon1<lon2: dx=-dx
    return [dx,dy]
  
def haversine(lat1,lat2,lon1,lon2):
    ###################################################
    # Great-circle distance between two lat/lon pairs #
    ###################################################
    # Assumes sea level #
    dlon=math.radians(lon2)-math.radians(lon1)
    dlat=math.radians(lat2)-math.radians(lat1)
    a=math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c=2*math.atan2(math.sqrt(a),math.sqrt(1-a))
    d=6373.0*c
    return d

def formEffectiveStressTensor(stresses,pressure):
    #####################################################################
    # Forms 3x3 effective stress tensor from pressure and four stresses #
    #####################################################################
    # order of vector: [xx,yy,xy,zz]
    ss=np.zeros([3,3])
    ss[0,0]=stresses[0]-pressure
    ss[0,1]=stresses[2]
    ss[1,0]=stresses[2]
    ss[1,1]=stresses[1]-pressure
    ss[2,2]=stresses[3]-pressure
    ss[2,0:2]=0. # Sxz and Syz
    ss[0:2,2]=0. # Szx and Szy - plain strain
    return ss
  
def projectStress(ss,nf,tf1,tf2): 
    #########################################
    # Projects stress onto fault plane slip #
    #########################################
    # ss - input stress tensor
    # nf - 
    tr=np.matmul(ss,nf)
    sn=np.sum(tr*nf)
    st1=np.sum(tr*tf1)
    st2=np.sum(tr*tf2)
    return (tr,sn,st1,st2)
  
def stressInvariants(Sxx,Syy,Szz,Sxy,Sxz,Syz,mu_r):
    #Compute stress invaraints
    #Output: mean, deviatoric, and MC 
    #Input: total or effective stress tensor components (scalar or matrix(node, time))
    #        rock internal friction 
    mean=(Sxx + Syy + Szz)/3.
    dev=np.sqrt(((Sxx-Syy)**2. + (Sxx-Szz)**2. + (Syy-Szz)**2. + 6.*(Sxy**2. + Sxz**2. + Syz**2.) )/6. )
    MC=dev - np.sin(np.arctan(mu_r))*mean
    return [mean,dev,MC]