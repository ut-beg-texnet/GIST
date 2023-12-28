#!/usr/bin/python
# Copyright 2022 ExxonMobil Upstream Research Company
# Authors: Prototype - Lei Jin; Port - Bill Curry
# Package Depenencies - pandas, scipy, numpy
# module load gcc/11.2.0 py-pandas py-scipy

# Potential Earthquake Scenario Toolkit
import gistMC as gi
import numpy as np
import pandas as pd

########################################
# Initialize GIST for shallow interval #
########################################
shallowM=gi.gistMC(minZ=3000.,maxZ=9000.,nReal=200)

shallowM.initPP(rho_0=1000.,nta=0.001,phi=15.,kMD=10.,h=4000.,alphav=1.e-9,beta=3.6e-10)

# To do - initialize these values to what gets matched from the earlier example
matchedNuUShallow=0.2852
shallowM.initPE(mu=20e9,nu=0.28,nu_u=matchedNuUShallow,alpha=0.31,mu_f=0.6,mu_r=0.6,match=False)

##################################################
# Initialize 2nd GIST instance for deep interval #
##################################################

deepM=gi.gistMC(minZ=10000.,maxZ=16000.,nReal=200)

deepM.initPP(rho_0=1000.,phi=6.6,kMD=7.45,h=1500.,alphav=1.e-9,beta=3.6e-10)

matchedNuUDeep=0.28098
deepM.initPE(mu=20e9,nu=0.28,nu_u=matchedNuUDeep,alpha=0.31,mu_f=0.6,mu_r=0.6,match=False)

# From previous run - scalar=12.)

#######################################
# Add uncertainties to both intervals #
#######################################

shallowM.initMCPP(rhoUnc=2.,ntaUnc=1.e-6,phiUnc=3.,kMDUnc=3.,hUnc=400.,alphavUnc=1.e-11,betaUnc=1.e-12)
shallowM.initMCPE(muUnc=1e9,nuUnc=0.002,nu_uUnc=0.002,alphaUnc=0.05,muFUnc=0.05,muRUnc=0.05,match=False)

deepM.initMCPP(rhoUnc=2.,ntaUnc=1.e-6,phiUnc=3.,kMDUnc=3.,hUnc=400.,alphavUnc=1.e-11,betaUnc=1.e-12)
deepM.initMCPE(muUnc=1e9,nuUnc=0.0005,nu_uUnc=0.0005,alphaUnc=0.05,muFUnc=0.05,muRUnc=0.05,match=False)


######################
# Post diffusivities #
######################
print("Shallow Diffusivity="+str(shallowM.diffPPMax))
print("Deep Diffusivity="+str(deepM.diffPPMax))

#######################
# Well info (from B3) #
#######################
# Note - this has gone through moderate processing
# Texas only wells, about 2000 or so around Gardendale
# To-do: Reprocess full TX and NM well inventory

wells=pd.read_csv('/users/bcurry/GIST/out15day.csv',sep=',',header=0)

#########################################
# Add potential wells to each instance  #
#########################################
# Shallow - this will do the windowing and calculate diffusion distances
shallowM.addTXWells(wells,'')
# Deep
deepM.addTXWells(wells,'')

# Optionally do a sanity check on the input wells
#shallowM.checkWells("Shallow ")
#deepM.checkWells("Deep ")
#print("Total wells: "+str(wells.shape[0]))
#print("Shallow wells: "+str(shallowM.nw)," "+str(shallowM.wells.shape[0]))
#print("Deep wells: "+str(deepM.nw)," "+str(deepM.wells.shape[0]))

# Optionally output windowed wells to separate CSV files
#shallowM.wells.to_csv('/users/bcurry/PEST/gardendale/gardendale_shallow_wells.csv')
#deepM.wells.to_csv('/users/bcurry/PEST/gardendale/gardendale_deep_wells.csv')


# Input earthquake info file - TexNet CSV format
eqs=pd.read_csv('/users/bcurry/GIST/gardendale/gardendale_stanton_events.csv',sep=',',nrows=2)

# This next part is pretty kludgey - should separate into subroutines

eqs=eqs.rename(columns={'Nodal Plane 1 - Strike': 'Strike','Nodal Plane 1 - Dip': 'Dip'})

# Add one phantom earthquake
fakeEQ={
  'EventID': 'fake0000',
  'Origin Date': '10/31/2021',
  'Origin Time': '00:00:00',
  'Magnitude': 1.0,
  'Latitude (WGS84)': 32.3,
  'Latitude Error (km)': 0.5,
  'Longitude (WGS84)': -102.4,
  'Longitude Error (km)': 0.5,
  'Strike':90.,
  'Dip':90.
  }

eqs=eqs.append(fakeEQ,ignore_index=True)

# I should probably have a method to process EQ info here

# Fix date columns
eqs[['Origin Date']]=eqs[['Origin Date']].apply(pd.to_datetime)

nQ=eqs.shape[0]

#############################
# Impulse response matching #
#############################

# Version 1 - take numbers from central matched impulse response
#             and put uncertainties around those poroelastic values
# First attempt
# Call matchPE2PP



# Version 2 - take all realizations from the pore pressure case 
#             and match poroelastic values to them
# TO DO
# Call matchPE2PP

#########################
# Make impulse response #
#########################
# This should be improved upon.
# Right now I kludge this by tweaking the scalar at the initialization and then looking at the result.
# Future work should be automating this match
# All of these parameters were arbitrary
#rvec=np.linspace(100,20000,200)
#bpd=60000
#days=np.linspace(0,100,501)
#duration=100
#impRespDF=deepM.impulseResponse(bpd,duration,days,rvec,recalc=True)
#impRespDF['FSP_PP']=impRespDF['FSP_PP']
#impRespDF.to_csv('/users/bcurry/GIST/poroimpulseresponse100.csv')



# Scenario generation
# Make empty dateframes for the output of each interval and each type of physics
sDFw=pd.DataFrame()
dDFw=pd.DataFrame()
sDFpp=pd.DataFrame()
dDFpp=pd.DataFrame()
sDFpe=pd.DataFrame()
dDFpe=pd.DataFrame()

shallowM.writeRealizations('shallowMC.csv')
deepM.writeRealizations('deepMC.csv')


# Loop over earthquakes
for iQ in range(nQ):
  print("Earthquake ",iQ," of ",nQ)
  eq=eqs.iloc[iQ]
  print(" EQ info:",eqs['EventID'][iQ],eqs['Latitude (WGS84)'][iQ],eqs['Longitude (WGS84)'][iQ])
  # Get list of wells given the maximum
  shallowWells=shallowM.findWells(eq)
  # Shallow scenario
  print(" Before shallow pressure scenario")
  sWellPPScenarios=shallowM.runPressureScenarios(eqs.iloc[iQ],shallowWells)
  print(" After shallow pressure scenario")
  sWellPEScenarios=shallowM.runPoroelasticScenarios(eqs.iloc[iQ],shallowWells)
  print(" After shallow poroelastic scenario")
  
  #sWellListPoro=shallowM.calcWellsPoro(eqs.iloc[iQ])
  #sWellList=shallowM.calcWells(eqs.iloc[iQ])
  # Deep scenario
  deepWells=deepM.findWells(eq)
  dWellPPScenarios=deepM.runPressureScenarios(eqs.iloc[iQ],deepWells)
  print(" After deep pressure scenario")
  dWellPEScenarios=deepM.runPoroelasticScenarios(eqs.iloc[iQ],deepWells)
  print(" After deep poroelastic scenario")
  
  # Append to dataframes
  sDFw=sDFw.append(shallowWells)
  dDFw=dDFw.append(deepWells)
  sDFpp=sDFpp.append(sWellPPScenarios)
  dDFpp=dDFpp.append(dWellPPScenarios)
  sDFpe=sDFpe.append(sWellPEScenarios)
  dDFpe=dDFpe.append(dWellPEScenarios)
sDFw=sDFw.drop(columns=['Dates','BPDs','Days'])
dDFw=dDFw.drop(columns=['Dates','BPDs','Days'])

# Write out lists
sDFw.to_csv('wells_shallow.csv')
dDFw.to_csv('wells_deep.csv')
sDFpp.to_csv('pressure_shallow.csv')
dDFpp.to_csv('pressure_deep.csv')
sDFpe.to_csv('poroelastic_shallow.csv')
dDFpe.to_csv('poroelastic_deep.csv')