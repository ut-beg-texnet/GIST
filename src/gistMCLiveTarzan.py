#!/usr/bin/python
###################################################
# GIST - Geomechanical Injection Scenario Toolkit #
# Driver script                                   #
###################################################
# Written in Python 3.6 #
#################################################################
# Copyright 2022 ExxonMobil Technology and Engineering Company  #
# Authors: MATLAB Prototype:  Lei Jin    lei.jin@exxonmobil.com #
#          Python port:    Bill Curry bill.curry@exxonmobil.com #
# Depenencies:                       pandas, scipy, numpy, math #
#       injectionV3 package needed to preprocess injection data #
#################################################################
# Internal module dependencies: # module load gcc/11.2.0 py-pandas py-scipy

import gistMCLive as gi
#import injectionV3 as i3
import numpy as np
import pandas as pd

########################
# List of earthquakes  #
# TexNet-like csv file #
########################

eqInFile='/users/bcurry/GIST/MidlandEQs.csv'

####################################################
# Injection files processed by injectionV3 package #
####################################################
# Usage - run injection processing every morning #
#         point to updated files here            #
##################################################
# Deep wells (below producing interval) #
#########################################
DeepWellFile='/data/xom/seismicity/bcurry/GIST/testDeep10.csv'
###########################################################
# Deep injection - 3 columns well ID, Day number, and BPD #
###########################################################
DeepInjFile='/data/xom/seismicity/bcurry/GIST/testDeep10Reg.csv'
#######################################################
# Shallow wells, injection (above producing interval) #
#######################################################
ShallowWellFile='/data/xom/seismicity/bcurry/GIST/testShallow10.csv'
ShallowInjFile='/data/xom/seismicity/bcurry/GIST/testShallow10Reg.csv'
################################################
# Output prefix for realizations of parameters #
################################################
RealizationPrefix='/data/xom/seismicity/bcurry/GIST/MC'
#######################
# Local run directory #
#######################
RunDir='/data/xom/seismicity/bcurry/GIST/'

# Parameter listing:



########################
# Read earthquake file #
########################
filtEQs=pd.read_csv(eqInFile,sep=',')
########################
# Get number of events #
########################
nQ=filtEQs.shape[0]
#########################################
# Convert dates to Pandas datetime type #
#########################################
filtEQs[['Origin Date']]=filtEQs[['Origin Date']].apply(pd.to_datetime)

#########################################################################
# To-do: ################################################################
#    Incorporate stress maps where we don't have fault plane solutions  #
#    Incorporate spatially-variable parameters (perm, porosity, height) #
#########################################################################
# Initialize GIST for shallow interval #
########################################
# nReal - number of realizations - important - should probably limit to 1-2000
# ntBin - not important to user - will determine automatically
shallowM=gi.gistMC(nReal=200,ntBin=21)
##############################################################
# Initialize pore pressure model                             #
# Parameters mostly from Katie Smye, are rough at this point 
# These are the important ones that change a lot:
# phi - porosity (0-100 percent)
# kMD - permeability in millidarcies
# h - interval thickness in feet
# These other ones are more stable:
# rho_0 - fluid density in kg/m3
# nta   - viscosity in ____
# alphav - vertical compressibility in 1/Pa - need to check
# beta   - fluid compressibility
##############################################################
shallowM.initPP(rho_0=1020.,nta=0.001,phi=10.,kMD=5.,h=1500.,alphav=1.e-9,beta=3.6e-10)
######################################################
# These parameters need more vetting for Tarzan area #
######################################################
# Important, but harder to parameterize
# mu - Shear modulus
# nu Poisson's ratio
# Nu_u undraine poisson's ratio - optional ,only if match=False
# alpha

shallowM.initPE(mu=20e9,nu=0.28,nu_u=0.2852,alpha=0.31,mu_f=0.6,mu_r=0.6,match=True)

##################################################
# Initialize 2nd GIST instance for deep interval #
##################################################
deepM=gi.gistMC(nReal=200,ntBin=21)
##################################################################
# Initialize pore pressure model                                 #
# Parameters mostly from Katie Smye, are better than for shallow #
##################################################################
deepM.initPP(rho_0=1020.,phi=1.5,kMD=33.,h=900.,alphav=1.e-9,beta=3.6e-10)
######################################################
# These parameters need more vetting for Tarzan area #
######################################################
deepM.initPE(mu=20e9,nu=0.28,nu_u=0.28098,alpha=0.31,mu_f=0.6,mu_r=0.6,match=True)
#######################################
# Add uncertainties to both intervals #
#######################################
shallowM.initMCPP(rhoUnc=5.,ntaUnc=1.e-6,phiUnc=5.,kMDUnc=4.,hUnc=400.,alphavUnc=1.e-11,betaUnc=1.e-12)
shallowM.initMCPE(muUnc=1e9,nuUnc=0.002,nu_uUnc=0.002,alphaUnc=0.05,muFUnc=0.05,muRUnc=0.05,match=True)

deepM.initMCPP(rhoUnc=5.,ntaUnc=1.e-6,phiUnc=1.,kMDUnc=25.,hUnc=400.,alphavUnc=1.e-11,betaUnc=1.e-12)
deepM.initMCPE(muUnc=1e9,nuUnc=0.0005,nu_uUnc=0.0005,alphaUnc=0.05,muFUnc=0.05,muRUnc=0.05,match=True)


######################
# Post diffusivities #
######################
print("Shallow Diffusivity="+str(shallowM.diffPPMax))
print("Deep Diffusivity="+str(deepM.diffPPMax))

#######################
# Well info (from B3) #
#######################

#########################################
# Add potential wells to each instance  #
#########################################
print("Before adding shallow wells")
shallowM.addWells(ShallowWellFile,ShallowInjFile)
# Deep
print("Before adding deep wells")
deepM.addWells(DeepWellFile,DeepInjFile)
###########################################
# Scenario generation                     #
# Make empty dateframes for the output of # 
# each interval and each type of physics  #
###########################################
sDFw=pd.DataFrame()
dDFw=pd.DataFrame()
sDFpp=pd.DataFrame()
dDFpp=pd.DataFrame()
sDFpe=pd.DataFrame()
dDFpe=pd.DataFrame()
##########################
# Write out realizations #
##########################
shallowM.writeRealizations(RealizationPrefix+'shallow.csv')
deepM.writeRealizations(RealizationPrefix+'deep.csv')

#########################
# Loop over earthquakes #
#########################
for iQ in range(nQ):
  print("Earthquake ",iQ," of ",nQ)
  eq=filtEQs.iloc[iQ]
  print(" EQ info:",eq['EventID'],eq['Origin Date'],eq['Latitude (WGS84)'],eq['Longitude (WGS84)'])
  # Get list of wells given the maximum
  shallowWells,shallowInj=shallowM.findWells(eq)
  # Shallow scenario
  print(" Before shallow pressure scenarios")
  sWellPPScenarios=shallowM.runPressureScenarios(eq,shallowWells,shallowInj,verbose=1)
  print(" After shallow pressure scenarios")
  sWellPEScenarios=shallowM.runPoroelasticScenarios(eq,shallowWells,shallowInj,verbose=1)
  print(" After shallow poroelastic scenarios")
  
  # Deep scenario
  deepWells,deepInj=deepM.findWells(eq)
  print(" Before deep pressure scenarios")
  dWellPPScenarios=deepM.runPressureScenarios(eq,deepWells,deepInj,verbose=1)
  print(" After deep pressure scenarios")
  dWellPEScenarios=deepM.runPoroelasticScenarios(eq,deepWells,deepInj,verbose=1)
  print(" After deep poroelastic scenarios")
  sWellPPScenarios.to_csv(RunDir+'pressure_shallow_live_eq'+str(iQ)+'.csv')
  sWellPEScenarios.to_csv(RunDir+'poroelastic_shallow_live_eq'+str(iQ)+'.csv')
  dWellPPScenarios.to_csv(RunDir+'pressure_deep_live_eq'+str(iQ)+'.csv')
  dWellPEScenarios.to_csv(RunDir+'poroelastic_deep_live_eq'+str(iQ)+'.csv')
  # Append to dataframes
  sDFw=sDFw.append(shallowWells)
  dDFw=dDFw.append(deepWells)
  sDFpp=sDFpp.append(sWellPPScenarios)
  dDFpp=dDFpp.append(dWellPPScenarios)
  sDFpe=sDFpe.append(sWellPEScenarios)
  dDFpe=dDFpe.append(dWellPEScenarios)

# Write out lists
sDFw.to_csv(RunDir+'wells_shallow_live.csv')
dDFw.to_csv(RunDir+'wells_deep_live.csv')
sDFpp.to_csv(RunDir+'pressure_shallow_live.csv')
dDFpp.to_csv(RunDir+'pressure_deep_live.csv')
sDFpe.to_csv(RunDir+'poroelastic_shallow_live.csv')
dDFpe.to_csv(RunDir+'poroelastic_deep_live.csv')
