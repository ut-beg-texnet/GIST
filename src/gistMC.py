"""@package docstring
GIST - Geomechanical Injection Scenario Toolkit

Copyright 2023 ExxonMobil Technology and Engineering Company 

Authors: MATLAB Prototype:  Lei Jin    lei.jin@exxonmobil.com
         Python port:    Bill Curry bill.curry@exxonmobil.com

Written in Python 3.6

Depenencies:                       pandas, scipy, numpy, math
      injectionV3 package needed to preprocess injection data
"""
############################################
# Well function for pore pressure solution #
############################################
import scipy.special as sc

#################################################################
# Interpolator for volumes time series for poroelastic solution #
#################################################################
import scipy.interpolate as si

##################
# Base libraries #
##################
import numpy as np
import pandas as pd
import math
import gc

# List manipulation - probably not needed
#from itertools import compress

#################################
# Contains:                     #
#   Classes:                    #
#     gistMC - Monte Carlo GIST #
#       init                    #
#       initPP                  #
#       initPE                  #
#       initPPAniso             #
#       writeRealizations       #
#       addWells                #
#       findWells               #
#       runPressureScenarios    #
#       pressureScenario        #
#       runPressureGrid
#       pressureScenarioAniso   #
#       runPoroelasticScenarios #
#       poroelasticScenario     #
#       poroAttr                #
#   Subroutines:                #
#     calcPPVals                #
#     matchPE2PP                #
#     calcPEvals                #
#     haversineXY               #
#     haversine                 #
#     formEffectiveStressTensor #
#     projectStress             #
#     stressInvariants          #
#     prepInj
#################################


class gistMC:
  """
  Monte-Carlo GIST class
  Instantiate one GIST class per 'scenario'        #
  Scenarios - deep or shallow injection triggering #
            - different central parameterizations  #
  
  Contains
  #####################################
  # init          - initialize general class ########################
  # initPP        - initialize pore pressure modeling #
  # initPE        - initialize poroelastic modeling  #
  # initPPAniso   - initialize anisotropic pore pressure modeling #
  # writeRealizations - output csv of parameter sets           #
  ####################################################################
  # addWells   - load injection and well info from injectionV3 files #
  # findWells  - get list of potential wells for an earthquake       #
  ####################################################################
  # pressureImpulseResponse - Produce impulse response for testing   #
  # poroelasticImpulseResponse - 
  # runPressureScenarios    - run all pore pressure scenarios for EQ #
  # pressureScenario        - individual pore pressure modeling case #
  # runPressureGrid ---
  # runPoroelasticScenarios - run all poroelastic scenarios for EQ   #
  # poroelasticScenario     - individual poroelastic modeling case   #
  # poroAttr                - poroelastic disaggregation to wells    #
  ####################################################################
  """
  def __init__(self,epoch=pd.to_datetime('01-01-1970'),nReal=100,seed=42,ntBin=51):
    """
    Constructor for base class
    # Inputs:
    #   epoch: Reference for Day=0 - UNIX Epoch default  #
    #   nReal: Number of Monte-Carlo realizations        #
    #    seed: Integer for random number generation      #
    #          Useful to create reproducable results     #
    #   ntBin: Factor of time interpolation needed for   #
    #          numerical integration for poroelastic eqn #
    """
    # Constants #
    #############
    # Gravity #
    ###########
    self.g=9.81
    ##################
    # Set parameters #
    ##################
    self.ntBin=ntBin
    self.epoch=epoch
    self.nReal=nReal
    ######################################
    # Initialize random number generator #
    ######################################
    self.rng=np.random.default_rng(seed=seed)
    ##############################
    # Initialize well data frame #
    ##############################
    self.nw=0
    self.wells=pd.DataFrame(columns=['ID','InjectionWellId','APINumber','UICNumber','Basin','SurfaceHoleLatitude','SurfaceHoleLongitude','WellName','InjectionType','CompletedWellDepthClassification','InjectionStatus','StartDate','PermittedMaxLiquidBPD','PermittedIntervalBottomFt','PermittedIntervalTopFt'])
    ###################################
    # Initialize injection data frame #
    ###################################
    self.inj=pd.DataFrame(columns=['ID','BPD','Days'])
    ######################################################################
    # Generate vector of nReal x 17 (number of parameters) random floats #
    ######################################################################
    self.randomFloats=self.rng.random(size=(self.nReal,17))
    #####################################################
    # Set initialization status for different scenarios #
    #####################################################
    self.runPP=False
    self.runPE=False
    self.runPPAniso=False
    self.runPEAniso=False
    #######################################
    # To-do: error checking of parameters # 
    #######################################
  
  def initPP(self,rho0_min=980.,rho0_max=1020.,
             nta_min=0.9e-3,nta_max=1.1e-3,
             phi_min=5.,phi_max=20.,
             kMD_min=20.,kMD_max=250.,
             h_min=80.,h_max=200.,
             alphav_min=1.08e-9,alphav_max=1.1e-9,
             beta_min=3.5e-10,beta_max=3.7e-10):
    """
    # Initialize Pore Pressure modeling #
    ############################################################
    # Inputs: ##################################################
    #      rho0_min/max :  Fluid density              (kg/m^3) #
    #       phi_min/max :  Porosity                  (percent) #
    #       nta_min/max :  Fluid viscosity    (Pascal-seconds) #
    #       kMD_min/max :  Permeability         (millidarcies) #
    #         h_min/max :  Injection interval thickness (feet) #
    #    alphav_min/max :  Vertical compressibility     (1/Pa) #
    #      beta_min/max :  Fluid compressibility        (1/Pa) #
    ############################################################
    # Assumptions: ########
    #     Runs after init #
    ########################
    # Set input parameters #
    ########################
    """
    self.phi_min=phi_min
    self.phi_max=phi_max
    self.kMD_min=kMD_min
    self.kMD_max=kMD_max
    self.rho_min=rho0_min
    self.rho_max=rho0_max
    self.nta_min=nta_min
    self.nta_max=nta_max
    self.alphav_min=alphav_min
    self.alphav_max=alphav_max
    self.beta_min=beta_min
    self.beta_max=beta_max
    self.h_min=h_min
    self.h_max=h_max
    
    #########################
    # Get vectors of values #
    #########################
    self.rhoVec   =self.rho_min    + self.randomFloats[:,0]*(self.rho_max   -self.rho_min)
    self.ntaVec   =self.nta_min    + self.randomFloats[:,1]*(self.nta_max   -self.nta_min)
    self.phiVec   =self.phi_min    + self.randomFloats[:,2]*(self.phi_max   -self.phi_min)
    self.hVec     =self.h_min      + self.randomFloats[:,3]*(self.h_max     -self.h_min)
    self.alphavVec=self.alphav_min + self.randomFloats[:,4]*(self.alphav_max-self.alphav_min)
    self.betaVec  =self.beta_min   + self.randomFloats[:,5]*(self.beta_max  -self.beta_min)
    # This is a linear distribution from the minimum to maximum
    self.kMDVec   =self.kMD_min    + self.randomFloats[:,6]*(self.kMD_max   -self.kMD_min)
    # I think that this is a logarithmic distribution from the minimum to maximum
    #self.kMDVec   =self.kMD_min    + np.power(10.,(self.randomFloats[:,6]*(np.log10(self.kMD_max)-np.log10(self.kMD_min))))

    ##############################################################
    # Convert vectors to SI units, generate intermediate vectors #
    ##############################################################
    (phiFracVec,hMVec,kapM2Vec,SVec,KVec,TVec,diffPPVec,CVec)=calcPPVals(self.kMDVec,self.hVec,self.alphavVec,self.betaVec,self.phiVec,self.rhoVec,self.g,self.ntaVec)
    ############################
    # Store in class variables #
    ############################
    self.phiFracVec=phiFracVec
    self.hMVec=hMVec
    self.kapM2Vec=kapM2Vec
    self.SVec=SVec
    self.KVec=KVec
    self.TVec=TVec
    self.diffPPVec=diffPPVec
    self.diffPPMax=np.max(diffPPVec)
    self.CVec=CVec
    #
    # Set initialized Flag as done
    #
    self.runPP=True
    ########################################################################
    # To-do - error checking - bounds of parameters, unphysical rock, etc. #
    ########################################################################
    return
  
  def initPE(self,mu_min=19e9,mu_max=21e9,
             nu_min=0.23,nu_max=0.25,
             nu_u_min=0.28,nu_u_max=0.32,
             alpha_min=0.26,alpha_max=0.36,
             mu_f_min=0.55,mu_f_max=0.65,
             mu_r_min=0.55,mu_r_max=0.65,
             match=True,verbose=0):
    """
    ###################################
    # Initialize poroelastic modeling #
    ##############################################################
    # Inputs: ####################################################
    #     mu_min/max : Rock shear modulus              (Pascals) #
    #     nu_min/max : Rock drained Poisson's ratio   (unitless) #
    #   nu_u_min/max : Rock undrained Poisson's ratio (unitless) #
    #                 Does not get used if match=True
    #  alpha_min/max : Biot's coefficient             (unitless) #
    #   mu_f_min/max : Fault friction coefficient     (unitless) #
    #   mu_r_min/max : Rock friction coefficient      (unitless) #
    #          match : Force matching diffusivities    (boolean) #
    ##############################################################
    # Assumptions: ###################
    #     Runs after init and initPP #
    ##################################
    """
    if self.runPP==False: print("gistMCLive.initPE Error: initPP must be run first!")
    self.mu_min=mu_min
    self.mu_max=mu_max
    self.nu_min=nu_min
    self.nu_max=nu_max
    ###########################################################
    # This does not get used if match=True which it should be #
    ###########################################################
    self.nu_u_min=nu_u_min
    self.nu_u_max=nu_u_max
    
    self.alpha_min= alpha_min
    self.alpha_max= alpha_max
    #######################################################
    # Friction coefficientS (unitless)                    #
    # mu_f - used when computing CFF                      #
    # mu_r - only used when computing MC stress invariant #
    #######################################################
    self.muF_min=mu_f_min
    self.muF_max=mu_f_max
    self.muR_min=mu_r_min
    self.muR_max=mu_r_max
    
    ##################################
    # Get vectors of values          #
    # Use randomfloats from initMCPP #
    ##################################
    self.muVec    =self.mu_min    + self.randomFloats[:,7] *(self.mu_max   -self.mu_min)
    self.nuVec    =self.nu_min    + self.randomFloats[:,8] *(self.nu_max   -self.nu_min)
    self.nu_uVec  =self.nu_u_min  + self.randomFloats[:,9] *(self.nu_u_max -self.nu_u_min)
    self.alphaVec =self.alpha_min + self.randomFloats[:,10]*(self.alpha_max-self.alpha_min)
    self.muFVec   =self.muF_min   + self.randomFloats[:,11]*(self.muF_max-self.muF_min)
    self.muRVec   =self.muR_min   + self.randomFloats[:,12]*(self.muR_max-self.muR_min)
    ##############################################################
    # Convert vectors to SI units, generate intermediate vectors #
    ##############################################################
    (lamdaVec,lamda_uVec,BVec,diffPEVec,kappaVec)=calcPEVals(self.muVec,self.nuVec,self.nu_uVec,self.alphaVec,self.kapM2Vec,self.ntaVec)
    ############################
    # Store in class variables #
    ############################
    self.lamdaVec=lamdaVec
    self.lamda_uVec=lamda_uVec
    self.BVec=BVec
    self.diffPEVec=diffPEVec
    self.kappaVec=kappaVec
    # This needs cleanup to remove repeated code #
    if match:
      ####################################
      # If we match - nu_u Unc goes away #
      ####################################
      if verbose>0: print(" Monte Carlo poroelastic - matched diffusivities")
      for i in range(len(self.lamdaVec)):
        # I don't think that we need this loop - just operate on the vectors #
        #############################
        # Get new Lame's parameters #
        #############################
        lamda,lamda_u = matchPE2PP(self.muVec[i],self.nuVec[i],self.alphaVec[i],self.CVec[i])
        nu_u=lamda_u/(2.*(lamda_u+self.muVec[i]))
        self.lamdaVec[i]=lamda
        self.lamda_uVec[i]=lamda_u
        #######################################
        # Recompute undrained Poisson's ratio #
        #######################################
        self.nu_uVec[i]=nu_u
        ###########################################
        # Recompute Skempton's coefficient vector #
        ###########################################
        self.BVec[i]=3.*(self.nu_uVec[i]-self.nuVec[i])/(self.alphaVec[i]*(1.+self.nu_uVec[i])*(1.-2.*self.nuVec[i]))
        ############################################
        # Recompute poroelastic diffusivity vector #
        # This should be the same as the PP one    #
        ############################################
        self.diffPEVec[i]=(self.kapM2Vec[i])*(self.lamda_uVec[i]-self.lamdaVec[i])*(self.lamdaVec[i]+2.*self.muVec[i])/(self.ntaVec[i]*self.alphaVec[i]*self.alphaVec[i]*(self.lamda_uVec[i]+2.*self.muVec[i]))
      if verbose>0:
        print(" Monte Carlo poroelastic (matched) - B     min/max:",np.amin(self.BVec),np.amax(self.BVec))
        print(" Monte Carlo poroelastic (matched) -diffPE min/max:",np.amin(self.diffPEVec),np.amax(self.diffPEVec))
    else:
      #########################################################################
      # Don't recompute - validity of undrained Lame's parameter not checked! #
      #########################################################################
      self.lamdaVec=2.*self.muVec*self.nuVec/(1.-2.*self.nuVec)
      self.lamda_uVec=2.*self.muVec*self.nu_uVec/(1.-2.*self.nu_uVec)
      ##################################
      # Compute Skempton's coefficient #
      ##################################
      self.BVec=3.*(self.nu_uVec-self.nuVec)/(self.alphaVec*(1.+self.nu_uVec)*(1.-2.*self.nuVec))
      #####################################
      # Compute diffusivity (poroelastic) #
      ################################################
      # Shouldn't this match the pore pressure one ? #
      ################################################
      self.diffPEVec=(self.kapM2Vec)*(self.lamda_uVec-self.lamdaVec)*(self.lamdaVec+2.*self.muVec)/(self.ntaVec*self.alphaVec*self.alphaVec*(self.lamda_uVec+2.*self.muVec))
    self.runPE=True
    return

  def initPPAniso(self,kMDSlow_min=1.,kMDSlow_max=100.,
                  kFastkSlow_min=1.,kFastkSlow_max=5.,
                  azimuthDeg_min=0.,azimuthDeg_max=10.,
                  kOffDiagRatio_min=-1.,kOffDiagRatio_max=1.,
                  verbose=0):
    """
    #######################################################
    # Initialize Anisotropy for Pore Pressure modeling    #
    #######################################################
    # Inputs: ####################################################################
    #        kMDSlow_min/max :  Permeabilty in slow direction             (millidarcies) #
    #     kFastkSlow_min/max :  Ratio of fast to slow lateral permeablity (unitless, >1) #
    #     azimuthDeg_min/max :  Azimuth of fast direction           (degrees CCW from E) #
    #  kOffDiagRatio_min/max :  Off-diagonal xy Permeability Ratio     (unitless, abs<1) #
    ##############################################################################
    # Assumptions: ##########
    #     Runs after initPP #
    #########################
    # Set input parameters #
    ########################
    """
    if self.runPP==False: print("gistMCLive.initPPAniso Error: initPP must be run first!")
    self.kMDSlow_min=kMDSlow_min
    self.kMDSlow_max=kMDSlow_max
    self.kFastkSlow_min=kFastkSlow_min
    self.kFastkSlow_max=kFastkSlow_max
    # I don't think that i need this
    self.kMDFast_min=kMDSlow_min*kFastkSlow_min
    self.kMDFast_max=kMDSlow_max*kFastkSlow_max
    #
    self.azimuth_min=azimuthDeg_min
    self.azimuth_max=azimuthDeg_max
    self.kOffDiagRatio_min=min(kOffDiagRatio_min,-1.)
    self.kOffDiagRatio_max=max(kOffDiagRatio_max,1.)
    # All of this stuff needs to be done on a per realization basis
    self.kMDOffDiag_min=kOffDiagRatio_min*np.sqrt(self.kMDSlow_min*self.kMDFast_min)
    # form permeability tensor
    kMDTensor=np.array([[self.kMDSlow, self.kMDOffDiag],[self.kMDOffDiag,self.kMDFast]])
    # form rotation tensor
    azRad=np.deg2rad(azimuthDeg)
    ATensor=np.array([[np.sin(azRad), -np.cos(azRad)],[np.cos(azRad),np.sin(azRad)]])
    # rotate permeability tensor to NE coordinates
    kMDRotate=np.matmul(ATensor.T,np.matmul(kMDTensor,ATensor))
    (kapM2,K,TAniso,diffPP,TBar)=calcPPAnisoVals(kMDRotate,self.hM,self.alphav,self.beta,self.phi,self.rho,self.g,self.nta)
    self.kMDRotate=kMDRotate
    self.TAniso=TAniso
    self.kapAnisoM2=kapM2
    self.TBar=TBar
    #########################
    # Get vectors of values #
    #########################
    self.kMDSlowVec   = self.kMDSlow_min    + np.power(10.,(self.randomFloats[:,13]*(np.log10(self.kMDSlow_max)-np.log10(self.kMDSlow_min))))
    self.kMDFastVec   = self.kMDSlowVec * ( self.kFastkSlow_min + self.randomFloats[:,14]*(self.kFastkSlow_max-self.kFastkSlow_min))
    self.azimuthVec   = self.azimuth_min + self.randomFloats[:,15]*(self.azimuth_max - self.azimuth_min)
    self.kMDOffDiagVec=np.sqrt(self.kMDSlowVec*self.kMDFastVec)*self.kOffDiagRatio_min+self.randomFloats[:,16]*(self.kOffDiagRatio_max-self.kOffDiagRatio_min)

    # Force kFast to be larger than kSlow
    self.kFastkSlowCentral,self.kFastkSlowUnc=logSpace(self.kFastkSlow,self.kFastkSlowLogUnc,clip=1.,verbose=1)
    # kMDFast * kMDSlow must be greater than kMDOffDiag**2
    # This means that kOffDiagRatio needs to be within -1 and 1
    # Find minimum and maximum values of kMDOffDiag
    lowerOffDiagRatio=max(-1.,self.kOffDiagRatio-kOffDiagRatioUnc)
    upperOffDiagRatio=min(1.,self.kOffDiagRatio+kOffDiagRatioUnc)

    # Initialize arrays
    self.kMDTensor=np.zeros([2,2,self.nReal])
    self.TAnisoVec=np.zeros([2,2,self.nReal])
    self.kapAnisoM2Vec=np.zeros([2,2,self.nReal])
    self.TBarVec=np.zeros([self.nReal,1])
    self.KAnisoVec=np.zeros([2,2,self.nReal])
    self.diffAnisoPPVec=np.zeros([2,2,self.nReal])
    azRadVec=np.deg2rad(self.azimuthVec)
    # Loop over realizations
    for iReal in range(self.nReal):
      # Form Rotation Tensor
      ATensor=np.array([[np.sin(azRadVec[iReal]), -np.cos(azRadVec[iReal])],[np.cos(azRadVec[iReal]),np.sin(azRadVec[iReal])]])
      kMDOrig=np.array([[self.kMDSlowVec[iReal], self.kMDOffDiagVec[iReal]],[self.kMDOffDiagVec[iReal],self.kMDFastVec[iReal]]])
      self.kMDTensor[:,:,iReal]=np.matmul(ATensor.T,np.matmul(kMDOrig,ATensor))
      (self.kapAnisoM2Vec[:,:,iReal],self.KAnisoVec[:,:,iReal],self.TAnisoVec[:,:,iReal],self.diffAnisoPPVec[:,:,iReal],self.TBarVec[iReal])=calcPPAnisoVals(self.kMDTensor[:,:,iReal],self.hMVec[iReal],self.alphavVec[iReal],self.betaVec[iReal],self.phiVec[iReal],self.rhoVec[iReal],self.g,self.ntaVec[iReal])

    self.runPPAniso=True
    return
    
  def writeRealizations(self,filePath):
    """
    #######################################################
    # Output csv file of PP and PE parameter realizations #
    #######################################################
    # Inputs: #############################################
    #     filePath : full path + name of output .csv file #
    #######################################################
    # Assumptions: #########################
    #     Runs after initMCPP and initMCPE #
    ########################################
    """
    # Make vector of realization number #
    #####################################
    realization=np.arange(self.nReal)
    ###################################
    # Form dictionary of realizations #
    # Then turn it into a dataframe   #
    ###################################
    d={}
    d['realization']=realization
    if self.runPP:
      d['rho']=self.rhoVec
      d['nta']=self.ntaVec
      d['phi']=self.phiVec
      d['kMD']=self.kMDVec
      d['h']=self.hVec
      d['alphav']=self.alphavVec
      d['beta']=self.betaVec
      d['kapM2']=self.kapM2Vec
      d['S']=self.SVec
      d['T']=self.TVec
      d['K']=self.KVec
      d['diffPP']=self.diffPPVec
    if self.runPE:
      d['mu']=self.muVec
      d['nu']=self.nuVec
      d['nu_u']=self.nu_uVec
      d['alpha']=self.alphaVec
      d['muF']=self.muFVec
      d['muR']=self.muRVec
      d['lambda']=self.lamdaVec
      d['lambdaU']=self.lamda_uVec
      d['B']=self.BVec
      d['diffPE']=self.diffPEVec
      d['kappa']=self.kappaVec
    if self.runPPAniso:
      d['kMDxx']=self.kMDTensor[0,0,:].flatten()
      d['kMDyy']=self.kMDTensor[1,1,:].flatten()
      d['kMDxy']=self.kMDTensor[0,1,:].flatten()
      d['diffxx']=self.diffAnisoPPVec[0,0,:].flatten()
      d['diffyy']=self.diffAnisoPPVec[1,1,:].flatten()
      d['diffxy']=self.diffAnisoPPVec[0,1,:].flatten()
      d['t_bar']=self.TBarVec.flatten()
      d['azimuth']=self.azimuthVec.flatten()
      # Not including
      #    self.TAnisoVec
      #    self.kapAnisoM2Vec
      #    self.KAnisoVec
    ########################
    # Convert to dataframe #
    ########################
    outDataFrame=pd.DataFrame(data=d)
    ################
    # Write to CSV #
    ################
    outDataFrame.to_csv(filePath)
    return
  

  def addWells(self,wellFile,injFile,verbose=0):  
    """
    ##############################################
    # Input csv files of well and injection data #
    ##############################################
    # Inputs: ################################################
    #     wellFile : full path + name of well .csv file      #
    #      injFile : full path + name of injection .csv file #
    ##########################################################
    # Assumptions: #################################
    #     Runs after init, inputs from injectionV3 #
    #     - wells merged, filtered and regularized #
    ################################################
    """
    self.wellFile=wellFile
    self.injFile=injFile
    #####################
    # Read in well file #
    #####################
    self.wellDF=pd.read_csv(wellFile)
    self.nw=self.wellDF.shape[0]
    # Check sanity of injection file
    injWellCount=pd.read_csv(self.injFile,usecols=['ID']).nunique()
    # This part is probably very slow
    injWellDays=pd.read_csv(self.injFile,usecols=['Days'])
    injWellDayMin=injWellDays.min()
    injWellDayMax=float(injWellDays.max())
    injWellDDay=injWellDays[injWellDays>injWellDayMin].min()-injWellDayMin
    self.injDT=float(injWellDDay)
    self.injOT=float(injWellDayMin)
    self.injNT=1+int((injWellDayMax-injWellDayMin)/injWellDDay)
    if verbose>0:
      print(' gistMC.addWells: well file added with ',self.nw,' wells')
      print(' gistMC.addWells: well columns:',self.wellDF.columns)
      print(' gistMC.addWells: injection file has ',injWellCount[0],' unique wells')
      print(' gistMC.addWells: injection file first day: ',self.injOT)
      print(' gistMC.addWells: injection file last day: ',injWellDayMax)
      print(' gistMC.addWells: injection file day interval: ',self.injDT)
      print(' gistMC.addWells: injection file number of time samples: ',self.injNT)
    self.runAddWells=True
    return

  def findWells(self,eq,PE=False,responseYears=0.,verbose=0):
    """
    ###########################################################
    # Get subset of wells within contribution distance/time   #
    ################################################################################
    # Input:                                                                       #
    #               eq: Earthquake dictionary - TexNet csv output                  #
    #               PE: if using poroelasticity, multiply diffusion distances by 6 #
    #    responseYears: Duration of response in years. Extend the time of the well #
    #                   selection by this amount. NOT CURRENTLY USED               #
    ################################################################################
    # Outputs:                                        #
    #    consideredWells: dataframe of selected wells #
    #       ignoredWells: dataframe of ignored wells  #
    #              injDF: dataframe of injection data #
    ###################################################
    # Assumptions: ##################
    #     Runs after init, addWells #
    #################################
    """
    # Initialize arrays #
    #####################
    wellDistances=np.zeros([self.nw])
    diffusionDistances=np.zeros([self.nw])
    encompassingDays=np.zeros([self.nw])
    encompassingDiffusivity=np.zeros([self.nw])
    dxs=np.zeros([self.nw])
    dys=np.zeros([self.nw])
    ddRatios=np.zeros([self.nw])
    wellDurations=np.zeros([self.nw])
    ##########################################
    # Get uncertainty for the earthquake     #
    # this is one standard deviation so we   #
    # are not being as conservative possible #
    ##########################################
    eqUncert=math.sqrt(eq['LatitudeError']**2+eq['LongitudeError']**2)
    ############################################################################
    # Step 1: compute per-well diffusion distances and distances to earthquake #
    ############################################################################
    for iw in range(self.nw):
      #####################################################################
      # Get number of days from start of injection to the earthquake date #
      #####################################################################
      injectionDays=(pd.to_datetime(eq['Origin Date'])-pd.to_datetime(self.wellDF['StartDate'][iw])).days
      wellDurations[iw]=injectionDays/365.25
      ###########################################################################
      # Find diffusion distance for each well at that date                      #
      # Equation 2.13 in documentation - take maximum pore pressure diffusivity #
      # In the case of poroelastic stressing, increase distance by 6 X          #
      #   This factor needs to be automatically computed in the future          #
      ###########################################################################
      injectionDaysClip=max(injectionDays,0.)
      if PE:
        diffusionDistances[iw]=6.*0.001*np.sqrt(4.*np.pi*self.diffPPMax*injectionDaysClip*24*60*60)
      else:
        diffusionDistances[iw]=0.001*np.sqrt(4.*np.pi*self.diffPPMax*injectionDaysClip*24*60*60)
      ##############################################
      # Compute distances from wells to earthquake #
      ##############################################
      wellDistances[iw]=haversine(eq['Latitude'],self.wellDF['SurfaceHoleLatitude'][iw],eq['Longitude'],self.wellDF['SurfaceHoleLongitude'][iw])
      ######################################################
      # Compute encompassing dates - what time will the    #
      # diffusion front from this well pass the epicenter? #
      # This is relative to the earthquake date            #
      ######################################################
      # First calculate the number of days for the well's  #
      # pressure front to reach the epicenter relative to  #
      # the wells first injection date                     #
      #######################################################
      # To-do: include earthquake location uncertainty here #
      #######################################################
      injWellDaysToEpicenter=(wellDistances[iw]*wellDistances[iw] /(4. * np.pi * self.diffPPMax))/(24*60*60)
      ##############################################################
      # Now add that number of days to the well time to get a date #
      ##############################################################
      injWellDateAtEpicenter=pd.to_datetime(self.wellDF['StartDate'][iw]) + pd.DateOffset(days=injWellDaysToEpicenter)
      ###################################################
      # Then subtract off the date of the earthquake to #
      # get a number of days relative to the earthquake #
      ###################################################
      encompassingDays[iw]=(pd.to_datetime(eq['Origin Date'])-injWellDateAtEpicenter).days
      #############################################################
      # Compute encompassing diffusivity - the diffusivity needed #
      # for this well to have been included in the analysis.      #
      #############################################################
      encompassingDiffusivity[iw]=(wellDistances[iw]*wellDistances[iw])/(4. * np.pi * injectionDays*24*60*60)
      ################################################################
      # Get an approximate x and y distance for poroelastic modeling #
      # Will also be needed for anisotropic permeability in v2       #
      ################################################################
      [dxs[iw],dys[iw]]=haversineXY(eq['Latitude'],self.wellDF['SurfaceHoleLatitude'][iw],eq['Longitude'],self.wellDF['SurfaceHoleLongitude'][iw])
      #
      # Ratio of diffusion distance to EQ distance
      # Smaller numbers mean more potential for influence
      # (with constant parameters)
      if diffusionDistances[iw]==0.:
        ddRatios[iw]=-1
      else:
        if PE:
          ddRatios[iw]=6.*wellDistances[iw]/diffusionDistances[iw]
        else:
          ddRatios[iw]=wellDistances[iw]/diffusionDistances[iw]
    ##############################################################
    # Step 2: Select wells where diffusion distances are greater #
    #         than the distance to the earthquake + uncertainty  #
    ##############################################################
    # To-do- Change this distance criteria to a time # 
    # criteria where we include responseYears        #
    ################################################## 
    consideredWellsDF=self.wellDF[diffusionDistances>(wellDistances-eqUncert)].reset_index(drop=True)
    consideredWellsDF['Distances']=wellDistances[diffusionDistances>(wellDistances-eqUncert)]
    consideredWellsDF['DXs']=dxs[diffusionDistances>(wellDistances-eqUncert)]
    consideredWellsDF['DYs']=dys[diffusionDistances>(wellDistances-eqUncert)]
    consideredWellsDF['DDRatio']=ddRatios[diffusionDistances>(wellDistances-eqUncert)]
    # Add a column with the time injecting prior to the earthquake
    consideredWellsDF['YearsInjecting']=wellDurations[diffusionDistances>(wellDistances-eqUncert)]
    consideredWellsDF['EncompassingDay']=encompassingDays[diffusionDistances>(wellDistances-eqUncert)]
    consideredWellsDF['EncompassingDiffusivity']=encompassingDiffusivity[diffusionDistances>(wellDistances-eqUncert)]
    ################################################################################
    # Create dataframe of wells that are ignored - this needs to be output as a QC #
    ################################################################################
    excludedWellsDF=self.wellDF[diffusionDistances<(wellDistances-eqUncert)].reset_index(drop=True)
    excludedWellsDF['Distances']=wellDistances[diffusionDistances<(wellDistances-eqUncert)]
    excludedWellsDF['DXs']=dxs[diffusionDistances<(wellDistances-eqUncert)]
    excludedWellsDF['DYs']=dys[diffusionDistances<(wellDistances-eqUncert)]
    excludedWellsDF['DDRatio']=ddRatios[diffusionDistances<(wellDistances-eqUncert)]
    # Add a column with the time injecting prior to the earthquake
    excludedWellsDF['YearsInjecting']=wellDurations[diffusionDistances<(wellDistances-eqUncert)]
    excludedWellsDF['EncompassingDay']=encompassingDays[diffusionDistances<(wellDistances-eqUncert)]
    excludedWellsDF['EncompassingDiffusivity']=encompassingDiffusivity[diffusionDistances<(wellDistances-eqUncert)]
    ##########################################################################
    # Step 3: Pull injection data from injection file that matches well list #
    #         and calculate total injected volume for all wells at EQ date   #
    ##########################################################################
    # Get list of IDs #
    ###################
    ids=consideredWellsDF['ID']
    excludedIDs=excludedWellsDF['ID']
    #####################
    # Open self.injFile #
    #####################
    iter_csv=pd.read_csv(self.injFile, iterator=True,chunksize=100000)
    #############################################################
    # One line - iteratively read through file and select wells #
    # Shelly suggests looking at SPARKF for reading through this #
    #############################################################
    #injDF= pd.concat([chunk[chunk['ID'].isin(ids)] for chunk in iter_csv])
    injDF=pd.DataFrame()
    injExcludedDF=pd.DataFrame()
    for chunk in iter_csv:
      # Collect injection data for selected wells
      injDF=pd.concat([injDF, chunk[chunk['ID'].isin(ids)]])
      # Collect injection data for unselected wells
      injExcludedDF=pd.concat([injExcludedDF, chunk[chunk['ID'].isin(excludedIDs)]])
    ############################################################
    # Get the number of wells selected from the injection file #
    ############################################################
    numDataWells=len(pd.unique(injDF['ID']))
    numExcludedWells=len(pd.unique(injExcludedDF['ID']))
    if verbose>0: print(" gistMC.findWells included: ",consideredWellsDF.shape[0],numDataWells,len(ids))
    if verbose>0: print(" gistMC.findWells excluded: ",excludedWellsDF.shape[0],numExcludedWells,len(excludedIDs))
    totalVolumes=np.zeros([len(ids)])
    # Loop over injection volume 
    for iwSelect in range(len(ids)):
      wellID=ids[iwSelect]
      if sum(injDF['ID']==wellID)==0:
        totalVolumes[iwSelect]=0.
      else:
        totalVolumes[iwSelect]=self.injDT*sum(injDF['BPD'][injDF['ID']==wellID])
    consideredWellsDF['TotalBBL']=totalVolumes
    totalExcludedVolumes=np.zeros([len(excludedIDs)])
    # Loop over injection volume 
    for iwSelect in range(len(excludedIDs)):
      wellID=excludedIDs[iwSelect]
      if sum(injExcludedDF['ID']==wellID)==0:
        totalExcludedVolumes[iwSelect]=0.
      else:
        totalExcludedVolumes[iwSelect]=self.injDT*sum(injExcludedDF['BPD'][injExcludedDF['ID']==wellID])
    excludedWellsDF['TotalBBL']=totalExcludedVolumes
    if verbose>0:
      ##########################################
      # Print total number of wells selected   #
      # and wells that have reported injection #
      ##########################################
      print(' gistMC.findWells: ',consideredWellsDF.shape[0],' wells considered')
      print(' gistMC.findWells: ',numDataWells,' wells with reported volumes, with ',injDF.shape[0],' injection values')
    return consideredWellsDF,excludedWellsDF,injDF
    
  def runPressureScenarios(self,eq,consideredWells,injDF,verbose=0):
    """
    ###########################################
    # Run all Monte Carlo pore pressure cases #
    ###########################################
    # Inputs: ##############################################
    #                 eq: earthquake dictionary            #
    #    consideredWells: dataframe of selected wells      #
    #              injDF: dataframe of injection for wells #
    #################################################################
    # Outputs:  scenarios: dataframe of scenarios split by operator #
    # Assumptions: ##################################################
    #     Runs after initMCPP, findWells                            #
    #     Faster than runPressureScenariosTimeSeries but only works #
    #     at the earthquake time                                    #
    #################################################################
    """
    ######################################################
    # Convert earthquake origin date to days since epoch #
    ######################################################
    eqDay=(pd.to_datetime(eq['Origin Date'])-self.epoch).days
    
    #nd=np.c(eqDay-self.injOT)
    #######################################################
    # Post number of wells considered for this earthquake #
    #######################################################
    nwC=consideredWells.shape[0]
    if verbose>0: print(" gistMC.runPressureScenarios: Number of wells considered: ",nwC)
    #################################
    # Initialize MC pressure arrays #
    #################################
    pressures=np.zeros([nwC,self.nReal])
    percentages=np.zeros([nwC,self.nReal])
    totalPressures=np.zeros([nwC,self.nReal])
    ####################################
    # Loop over wells in consideration #
    # Secondary loop is realizations   #
    ####################################
    for iwc in range(nwC):
      if verbose>0 and iwc%10==0: print(" gistMC.pressureScenario: well ",iwc," of ",nwC,": ",consideredWells['ID'][iwc])
      #################################
      # Injection rates for this well #
      #################################
      try:
        wellID=consideredWells['ID'][iwc].tolist()[0]
      except:
        try:
          wellID=consideredWells['ID'][iwc].tolist()
        except:
          print('ID ',iwc)
      bpds=injDF['BPD'][injDF['ID']==wellID]
      ################################
      # Injection days for this well #
      ################################
      days=injDF['Days'][injDF['ID']==wellID]
      #################################
      # Convert distance from km to m #
      #################################
      try:
        dist=np.real(1000.*consideredWells['Distances'][iwc].tolist()[0])
      except:
        dist=np.real(1000.*consideredWells['Distances'][iwc].tolist())
      ##########################
      # Loop over realizations #
      # to model pressures     #
      ##########################
      # This is be pulled out of two loops and vectorized elsewhere #
      for iReal in range(self.nReal):
        pressure=self.pressureScenario(bpds,days,eqDay,dist,iReal)
        pressures[iwc,iReal]=pressure
    ##################################
    # Form dataframe of realizations #
    ##################################
    scenarios=pd.DataFrame(columns=['EventID','EventLatitude','EventLongitude','ID','Name','API','Latitude','Longitude','NumWells','Pressures','TotalPressure','Percentages','Realization'])
    ###############################
    # Loop over realizations      #
    # for disaggregation to wells #
    ###############################
    for iReal in range(self.nReal):
      scenarioDF=pd.DataFrame(columns=['EventID','EventLatitude','EventLongitude','ID','Name','API','Latitude','Longitude','NumWells','Pressures','TotalPressure','Percentages','Realization'])
      ######################################
      # Sum pressures for this realization #
      # and disaggregate - equation 4.9    #
      ######################################
      totalPressures[:,iReal]=np.sum(pressures[:,iReal])
      percentages[:,iReal]=100.*pressures[:,iReal]/totalPressures[:,iReal]
      ##################################################
      # Number of wells with a meaningful contribution #
      ##################################################
      nwCnz=sum(percentages[:,iReal]>0.01)
      ############################
      # Set columns of dataframe #
      ############################
      scenarioDF['Pressures']=pressures[:,iReal]
      scenarioDF['TotalPressure']=totalPressures[:,iReal]
      scenarioDF['Percentages']=percentages[:,iReal]
      scenarioDF['EventID']=eq['EventID']
      scenarioDF['EventLatitude']=eq['Latitude']
      scenarioDF['EventLongitude']=eq['Longitude']
      scenarioDF['API']=consideredWells['APINumber']
      scenarioDF['Name']=consideredWells['WellName']
      scenarioDF['ID']=consideredWells['ID']
      scenarioDF['Latitude']=consideredWells['SurfaceHoleLatitude']
      scenarioDF['Longitude']=consideredWells['SurfaceHoleLongitude']
      scenarioDF['NumWells']=nwCnz
      ##########################
      # Add realization number #
      ##########################
      scenarioDF['Realization']=iReal
      #######################
      # Append to scenarios #
      #######################
      scenarios=pd.concat([scenarios,scenarioDF],ignore_index=True)
      if verbose>0: print(" gistMC.pressureScenario: scenario ",iReal+1," of ",self.nReal,": Max P:",max(pressures[:,iReal]))
    ####################
    # Return scenarios #
    ####################
    return scenarios

  
  def runPressureScenariosTimeSeries(self,eq,consideredWells,injDF,verbose=0):
    """
    ###############################################################################
    # runPressureScenariosTimeSeries:                                             #
    #        Version of pore pressure modeling to output time series of pressures.#
    #        This will be slower than runPressureScenarios which only puts out    #
    #        pressures at the EQ time at one time.                                #
    ###############################################################################
    # Inputs:                                                                    #
    #        eq:               earthquake dataframe with 'Origin Date' column    #
    #        consideredWells:  dataframe of wells produced by self.findWells     #
    #                          with 'ID' and 'Distances' columns                 #
    #        injDF:            dataframe of injection produced by self.findWells #
    #                          with 'ID', 'Days' and 'BPD' columns               #
    ##############################################################################
    # Outputs:                                                                   #
    #        scenarioDF:       dataframe of pore pressure contribution scenarios #
    #                          with many columns at earthquake date              #
    #        dP:               pressure time series at eq location               #
    #                          nwC x nReal is pretty big                         #
    #                          size(nwC x nReal x nt)                            #
    ##############################################################################
    # To-do:  Optionally give a list of r values to compute on a grid #
    #         Currently implemented in runPressureGrid                #
    ###################################################################
    """
    eqDay=(pd.to_datetime(eq['Origin Date'])-self.epoch).days
    ########################################################################
    # Prep injection data to get arrays needed for vectorized calculations #
    ########################################################################
    (wellIDs,nwC,dayVec,nt,ot,bpdArray,secArray,dx,dy,wellDistances,ieq,feq)=prepInj(consideredWells,injDF,self.injDT,dxdyIn=None,eqDay=eqDay,endDate=None)
    if verbose>0: print('runPressureScenariosTimeSeries time axis information - nt:',nt,'; ot:',ot,'; dt:',self.injDT,' earthquake index: ',ieq)
    #######################################################
    # Floating point linear interpolation factor - scalar #
    #######################################################
    f=feq-ieq
    ###########################################
    # Convert bpdArray to Q - m3/s [nt+1,nwC] #
    ###########################################
    QArray=1.84013e-6 *bpdArray
    #######################################################
    # Take a derivative of QArray along the time (0) axis #
    # This array now has one fewer time samples [nt,nwC]#
    #######################################################
    dQdtArray=np.diff(QArray,axis=1)
    ##############################################
    # Compute r squared for all wells [nwC] #
    ##############################################
    r2=wellDistances*wellDistances
    if verbose>0: print('runPressureScenariosTimeSeries r2 min/max: ',min(r2),max(r2))
    #####################################################
    # Compute property-related part of ppp [nReal] #
    #####################################################
    TSOver4TT=self.TVec*self.SVec/(4.*self.TVec*self.TVec)
    if verbose>0: print('runPressureScenariosTimeSeries TSOver4TT min/max: ',min(TSOver4TT.flatten()),max(TSOver4TT.flatten()))
    #######################################################################
    # Compute outer product of r2 and TSOver4TT to get ppp [nwC,nReal] #
    #######################################################################
    ppp=np.outer(r2,TSOver4TT)
    if verbose>0: print('runPressureScenariosTimeSeries ppp min/max: ',min(ppp.flatten()),max(ppp.flatten()))
    #############################
    # Compute gRhoOverT [nReal] #
    #############################
    gRhoOverT=self.rhoVec*self.g/(4.*np.pi*self.TVec)
    ########################
    # Initialize output dP #
    ########################
    dP=np.zeros([nwC,self.nReal,nt])
    ######################################
    # Convert injDT from days to seconds #
    ###################################### 
    dts=self.injDT*24*60*60
    #######################################################################################
    # Create a vector of injection durations starting with all time and ending with dt.   #
    # Variable-injection Theis modeling sums a shortening series of boxcars with          #
    # different heights corresponding to changes in injection rates over time - dQdtArray #
    #######################################################################################
    durations=np.max(secArray)-secArray+dts
    if verbose>0: print('runPressureScenariosTimeSeries durations min/max: ',min(durations),max(durations))
    #######################################################################################
    # This has the well function in it - sc.exp1. Moving this out of the loop speeds up   #
    # computation vs. FSP for a time series by O(nt). epp is [nwC,nReal,nt].              #
    # We reuse parts of this array in the summation as we assume that dt is fixed.        #
    # I'm sure that there are better ways to broadcast these shapes but I don't know how! #
    #######################################################################################
    epp=sc.exp1(ppp.reshape((nwC,self.nReal,1)).repeat(nt,2) / durations[:].reshape((1,1,nt)).repeat(nwC,0).repeat(self.nReal,1))
    if verbose>0: print('runPressureScenariosTimeSeries epp min/max: ',min(epp.flatten()),max(epp.flatten()))
    ##########################
    # Loop over output time: #
    ##########################
    for it in range(1,nt):
      if verbose>0:
        if mod(it,10)==0: print('runPRessureScenariosTimeSeries time step ',it,' of ',nt)
      #########################################################
      # Get output of well function x the change in injection #
      # We take the last 'it' values of epp that is a series  #
      # of boxcars ranging from it*dt to dt and dot product   #
      # it with the first 'it' values of the dQdtArray to get #
      # the time series output at 'it'. This is really a      #
      # convolution of epp and dQdtArray on the last axis and #
      # there should be a way to make this more efficient!    #
      #########################################################
      timeStepsSum=np.sum(epp[:,:,-it:] * dQdtArray[:,:it].reshape((nwC,1,it)).repeat(self.nReal,1),axis=2)
      ########################################################################
      # Multiply the sum of the time steps with gRhoOverT and convert to PSI #
      # dP is the change in pressure from the first time step [nw,nReal,nt]  #
      ########################################################################
      dP[:,:,it]=timeStepsSum * gRhoOverT.reshape((1,self.nReal)).repeat(nwC,0) / 6894.76
    ###################################################
    # Linear interpolation between the two time steps #
    # bounding the EQ time dPatEQ [nw,nReal]          #
    ###################################################
    dPatEQ=((1.-f)*dP[:,:,ieq])+(f*dP[:,:,ieq+1])
    ###########################################################
    # Sum over wells to get total Pressure at EQ time [nReal] #
    ###########################################################
    totalPressureAtEQ=np.sum(dPatEQ,axis=0,keepdims=True)
    #########################################################
    # Calculate percentages for each realization [nw,nReal] #
    #########################################################
    percentages=100.* dPatEQ / totalPressureAtEQ.repeat(nwC,0)
    ##########################################
    # Get dataframe of output scenarios from #
    # input numpy arrays and well dataframe  #
    ##########################################
    scenarioDF=self.pressureScenariosToDF(eq,consideredWells,dPatEQ,totalPressureAtEQ,percentages)
    return scenarioDF,dP,wellIDs,dayVec

  def runPressureGrid(self,wellDF,injDF,grid,dt=10,verbose=0):
    '''
    ###############################################################
    # runPressureGrid: Create grid of pressures for visualization #
    ###############################################################
    # Inputs: ######################################
    #   wellDF: Dataframe of well information      #
    #   injDF:  Dataframe of injection information #
    #   grid:   grid defintion [nx,ny,2]           #
    #     dt:   time sampling of injDF             #
    ################################################
    # Outputs: #####################################
    #   Pressures                [nx,ny,nw,nReal,nt]
    ################################################
    # Warning: this can fill memory if you do too  #
    #    many realizations, wells, or time steps!  #
    ################################################
    # To-do: produce output summed over wells      #
    ################################################
    '''
    ########################################################################
    # Prep injection data to get arrays needed for vectorized calculations #
    ########################################################################
    (wellIDs,nwC,dayVec,nt,ot,bpdArray,secArray,dx,dy,wellDistances,dxdyGrid)=prepInj(wellDF,injDF,dt,dxdyIn=grid,eqDay=None,endDate=None)
    nw=wellDF.shape[0]
    nx=grid.shape[0]
    ny=grid.shape[1]
    nxy=nx*ny
    nwxy=nw*nx*ny
    nt=len(secArray)
    distances=np.zeros([nx,ny,nw])
    dts=dt*24*60*60
    #########################################################
    # Calculate distances from each well to all grid points #
    #########################################################
    for iw in range(nw):
      distances[:,:,iw]=np.sqrt((dxdyGrid[0,:,:,0]*dxdyGrid[0,:,:,0]) + (dxdyGrid[0,:,:,1]*dxdyGrid[0,:,:,1]))
    ##########################################################################
    # This is almost identical to the runPressureScenariosTimeSeries         #
    # Similar vectorization, but evaluating at grid points, not EQ locations #
    # Look at above comments, I should standardize this code and not repeat! #
    ##########################################################################
    QArray=1.84013e-6 *bpdArray
    dQdtArray=np.diff(QArray,axis=1)
    #########################################################
    # compute r squared for all wells to all grid locations #
    # [nx x ny x nw,] flattened to a vector                 #
    #########################################################
    r2=np.ravel(distances*distances)
    TSOver4TT=self.TVec*self.SVec/(4.*self.TVec*self.TVec)
    ppp=np.outer(r2,TSOver4TT)
    gRhoOverT=self.rhoVec*self.g/(4.*np.pi*self.TVec)
    ############################################################################
    # This output dP is nx x ny larger than in runPressureScenariosTimeSeries! #
    ############################################################################
    dP=np.zeros([nx,ny,nw,self.nReal,nt])
    durations=np.max(secArray)-secArray+dts
    epp=sc.exp1(ppp.reshape((nwxy,self.nReal,1)).repeat(nt,2) / durations[:].reshape((1,1,nt)).repeat(nwxy,0).repeat(self.nReal,1)).reshape((nx,ny,nw,self.nReal,nt))
    for it in range(1,nt):
      timeStepsSum=np.sum(epp[:,:,:,:,-it:] * dQdtArray[:,:it].reshape((1,1,nw,1,it)).repeat(self.nReal,3),axis=4)
      dP[:,:,:,:,it]=timeStepsSum.reshape(nx,ny,nw,self.nReal) * gRhoOverT.reshape((1,1,1,self.nReal)).repeat(nw,2).repeat(nx,0).repeat(ny,1) / 6894.76
    ##############
    # dP is big! #
    ##############
    return dP

  def pressureScenario(self,bpds,days,eqDay,r,iReal):
    """
    ###################################
    # Pore pressure modeling a la FSP #
    ###################################
    # Inputs: ################################################
    #            bpds: list of injection rates (Barrels/day) #
    #            days: list of injection days                #
    #           eqDay: day number of earthquake              #
    #               r: distance to earthquake       (meters) #
    #           iReal: realization number                    #
    ##########################################################
    # Outputs:      dP: modeled change in pressure      (PSI) # 
    # Assumptions: #########################################
    #     Run inside runPressureScenarios #
    # To-do - replace with Lei's code with additional terms #
    """
    #######################################
    # Count number of time steps #
    ##############################
    nd=np.count_nonzero(days<eqDay)
    ########################################################
    # If the earthquake was before any injection, return 0 #
    ########################################################
    if nd==0: return 0.
    # Get the starting time step #
    od=int((min(days)-self.injOT)/self.injDT)
    # Get the last time step #
    ed=int((max(days)-self.injOT)/self.injDT)
    # Get the number of days for this well
    ndOut=ed-od+1
    ##########################################
    # Array of time in seconds to earthquake #
    ##########################################
    priorDays=days[days<eqDay].to_list()
    # This needs to be recomputed for each output time, not just the earthquake time!
    priorSec=[24*60*60*(eqDay-day) for day in priorDays]
    ##############################################
    # Array of cubic meters per second injection #
    ##############################################
    #########################################################
    # 1 oil barrel per day = 1.84013e-6 cubic meters/second #
    # according to Rall's Google search                     #
    #########################################################
    priorBPD=bpds[days<eqDay].to_list()
    priorQ=[1.84013e-6 * bpd for bpd in priorBPD] 
    ##############################
    # Initialize timeSteps array #
    ##############################
    timeSteps=np.zeros([ndOut,])
    # Precompute part of PP
    ppp=(r*r*self.TVec[iReal]*self.SVec[iReal])/(4.*self.TVec[iReal]*self.TVec[iReal])
    ###################
    # First time step #
    ###################
    timeSteps[0]=sc.exp1(ppp/(priorSec[0]))*priorQ[0]
    ########################
    # Loop over time steps #
    ########################
    for i in range(1,nd):
      ###########################################
      # Equation 1.16 - pp is the argument of W #
      # This is problematic for outputing at any time other than the EQ time
      ###########################################
      pp = ppp/(priorSec[i])
      #####################################
      # Well function W from Equation 1.1 #
      #####################################
      wellFunc = sc.exp1(pp)
      #######################################################
      # Multiplication by (Qt - Qt-1) term in Equation 1.16 #
      #######################################################
      timeSteps[i] = wellFunc*(priorQ[i]-priorQ[i-1])
    ######################################################################
    # Summation over time steps and scaling of result from Equation 1.16 #
    ######################################################################
    head = sum(timeSteps)*(1./(4.*np.pi*self.TVec[iReal]))
    #################################################
    # Pressure change from head * density * gravity #
    #################################################
    dP = head*self.rhoVec[iReal]*self.g
    ##################
    # Convert to PSI #
    ##################
    dP=dP/6894.76
    #####################################################
    # Sanity check for negative pressure!               # 
    # I had an off-by-one bug earlier I caught this way #
    #####################################################
    if dP<0.: print(" gistMC.pressureScenario: Negative pressure! ",dP,",",timeSteps)
    return dP

  def runPressureScenariosAniso(self,eq,consideredWells,injDF,endDate=None,dxdy=None,verbose=0):
    """
    pressureScenarioAniso: pore pressure modeling with anisotropy
    Inputs: 
            eq:               earthquake dataframe with 'Origin Date' column
            consideredWells:  dataframe of wells produced by self.findWells
                              with 'ID' and 'Distances' columns
            injDF:            dataframe of injection produced by self.findWells
                              with 'ID', 'Days' and 'BPD' columns 
            dxdy:             OPTIONAL numpy[nd,2] array of x,y distances from 
                              earthquake to evaluate
    Outputs:
            scenarioDF:       dataframe of pore pressure contribution scenarios
                              with many columns at earthquake date
            dP:               pressure time series at eq location
                              nt x nwC x nReal is pretty big
                              size(nt x nwC x nReal)
    To-do:  optionally give a list of r values to compute on a grid
            for a single realization?
            - This code needs cleaning up, lots of duplicated work!
    """
    eqDay=(pd.to_datetime(eq['Origin Date'])-self.epoch).days
    # Go from the consideredWells and injDF dataframes to
    # numpy arrays of rates (nw,nt), well distances (2,nw), and times (nt)
    # This should be a separate subroutine
    (wellIDs,nwC,dayVec,nt,ot,bpdArray,secArray,dx,dy,wellDistances,ieq,feq) = prepInj(consideredWells,injDF,self.injDT,eqDay=eqDay,endDate=endDate)
    print('runPressuresAniso: Through setup')
    gRhoOverT=self.rhoVec*self.g/(4.*np.pi*self.TVec)

    # Now we have our 2D bpdArray of injection data
    # convert bpdArray to Q - m3/s
    QArray=1.84013e-6 *bpdArray
    # take a derivative of QArray along the time (0) axis
    # This is now shorter by 1 - don't need this for one value
    dQdtArray=np.diff(QArray,axis=0)
    # compute r squared for all wells - size nwC
    r2=wellDistances*wellDistances
    # Get X and Y values
    # Get 
    #Radial (in-plane) distance to well
    #R=sqrt((X-Wells.x).^2+(Y-Wells.y).^2); % distance from wells 
    #Rv=R(:); 
    r=np.sqrt(dx*dx+dy*dy)
    #Initialization
    nts=nt-1
    # hydraulic head, m, for method 1 
    H1=np.zeros([len(r),self.nReal,nts])
    # hydraulic head, m, for method 2
    #H2=np.zeros([len(r),self.nReal,nts])
    # left-end  time of each injection inteRval (ti-1 vector); must be specified in data if intervals not equal-time
    # to-do - remove this and have a single time since we have a regular time input
    ta=secArray-self.injDT
    # right-end time of each injection interval (ti vector)
    tb=secArray

    #% Intermeditate constants 
    # This constant is per-realization - size nReal
    cons2=1./(4.*np.pi*self.TBarVec)
    # Temporary array of size (nwC, nReal) -  precompute outside of loop
    TxR=(self.TAnisoVec[0,0,:].reshape((1,self.nReal,1)).repeat(nwC,0)  *(dy*dy).reshape((nwC,1,1)).repeat(self.nReal,1) + 
        self.TAnisoVec[1,1,:].reshape((1,self.nReal,1)).repeat(nwC,0)   *(dx*dx).reshape((nwC,1,1)).repeat(self.nReal,1) - 
        2.*self.TAnisoVec[0,1,:].reshape((1,self.nReal,1)).repeat(nwC,0)*(dx*dy).reshape((nwC,1,1)).repeat(self.nReal,1))
    # Now we are taking the array and making it nt-1 size so that we do no divide by zero
    dta=secArray[-1]-ta[:-1]
    dtb=secArray[-1]-tb[:-1]
    # u and v are size (nwC,nReal,nt-1)
    TBar2=self.TBarVec * self.TBarVec
    u=((0.25*self.SVec.reshape((1,self.nReal,1)).repeat(nwC,0) * TxR).repeat(nts,2) / 
       (TBar2.reshape((1,self.nReal,1)).repeat(nwC,0).repeat(nts,2) * dta.reshape(1,1,nts).repeat(nwC,0).repeat(self.nReal,1)))
    v=((0.25*self.SVec.reshape((1,self.nReal,1)).repeat(nwC,0) * TxR).repeat(nts,2) / 
       (TBar2.reshape((1,self.nReal,1)).repeat(nwC,0).repeat(nts,2) * dtb.reshape(1,1,nts).repeat(nwC,0).repeat(self.nReal,1)))
    #v=(0.25*self.SVec.reshape((1,self.nReal,1)).repeat(nwC,0) * TxR).repeat(nt,2) / (self.TBarVec * self.TBarVec * dtb)
    print('runPressuresAniso: Before exp1')
    # To-do - only do this once, check for NaNs - we don't need a zero length tb
    Wa=sc.exp1(u)
    Wb=sc.exp1(v)
    print('runPressuresAniso: After exp1',QArray.shape)
    del u
    del v
    gc.collect()
    # This should be a time derivative
    WaWb=(Wa-Wb)*cons2.reshape(1,self.nReal,1).repeat(nwC,0).repeat(nts,2)
    del Wa
    del Wb
    gc.collect()
    print('runPressuresAniso: Before sum',len(QArray[:,:0]))
    # Loop over time
    for k in range(nts):
      print('runPressuresAniso: Time Step ',k,' of ',nts)
      # Select range of WaWb we need - lask k elements and multiply with first k elements of injection
      H1[:,:,k]=np.sum(QArray[:,:k+1].reshape((nwC,1,k+1)).repeat(self.nReal,1)*WaWb[:,:,-k-1:],2)
    print('runPressuresAniso: After sum')
    dP=self.g*self.rhoVec.reshape((1,self.nReal,1)).repeat(nwC,0).repeat(nts,2)*H1; 
    dP=dP/6894.76
    # linear interpolation of two time steps
    dPatEQ=((1.-feq)*dP[:,:,ieq])+(feq*dP[:,:,ieq+1])
    return dP, dPatEQ
  
  def runPoroelasticScenarios(self,eq,consideredWells,injDF,verbose=0):
    """
    #########################################
    # Run all Monte Carlo poroelastic cases #
    #########################################
    # Inputs: ##############################################
    #                 eq: earthquake dictionary (TexNet)   #
    #    consideredWells: dataframe of selected wells      #
    #              injDF: dataframe of injection for wells #
    ########################################################
    # Output:  scenarios: dataframe of scenarios           # 
    # Assumptions: #########################################
    #     Runs after initMEPP, findWells #
    ######################################################
    # To-do: pull stress information from a map if we    # 
    #      lack a focal plane, deal with ambiguity       #
    ######################################################
    """
    # Convert earthquake origin date to days since epoch #
    ######################################################
    eqDay=(pd.to_datetime(eq['Origin Date'])-self.epoch).days
    
    #######################################################
    # Post number of wells considered for this earthquake #
    #######################################################
    nwC=consideredWells.shape[0]
    if verbose>0: print(" gistMC.runPoroelasticScenarios: Number of wells considered: ",nwC)
    
    ############################################
    # Initialize MC pressure and stress arrays #
    ############################################
    # Stress is 4 values here: [xx,xy,yy,zz] #
    ##########################################
    pressures=np.zeros([nwC,self.nReal])
    stresses=np.zeros([nwC,4,self.nReal])
    totalPressure=np.zeros([nwC,self.nReal])
    totalStresses=np.zeros([nwC,self.nReal])
    thetaVec=np.zeros([nwC,self.nReal])
    poroStress=np.zeros([nwC,self.nReal])
    percentages=np.zeros([nwC,self.nReal])
    ###########################################################################################
    # To-do: Pull from a grid of stress data what the most likely earthquake is at a location #
    #        Also need to worry about nonuniqueness of stress and fault plane information     #
    ###########################################################################################
    # Backup in case strike and dip of earthquake are not specified #
    #################################################################
    if np.isnan(eq['Strike']):
      if verbose>0: print(' gistMC.runPoroelasticScenarios: EQ Strike not specified!')
      az=90.
    else:
      az=eq['Strike']
    if np.isnan(eq['Dip']):
      if verbose>0: print(' gistMC.runPoroelasticScenarios: EQ Dip not specified!')
      dip=90.
    else:
      dip=eq['Dip']
    ####################################
    # Loop over wells in consideration #
    # Secondary loop is realizations   #
    ####################################
    for iwc in range(nwC):
      if verbose>0 and iwc%10==0: print(" gistMC.runPoroelasticScenarios: well ",iwc," of ",nwC,": ",consideredWells['ID'][iwc])
      #################################
      # Injection rates for this well #
      #################################
      bpds=injDF['BPD'][injDF['ID']==consideredWells['ID'][iwc]]
      ################################
      # Injection days for this well #
      ################################
      days=injDF['Days'][injDF['ID']==consideredWells['ID'][iwc]]
      ##################################
      # Convert distances from km to m #
      ##################################
      dist=1000.*consideredWells['Distances'][iwc]
      dx=1000.*consideredWells['DXs'][iwc]
      dy=1000.*consideredWells['DYs'][iwc]
      #
      # Add additional check on inputs
      #
      if verbose>1:
        print('    runPoroelasticScenarios: well ',consideredWells['ID'][iwc],': ',dist,'m away, dx=',dx,', dy=',dy)
        if len(days)>0:
          print('      with ',len(days),' injection values from ',min(days),' to ',max(days),' max rate is ',max(bpds))
        else:
          print('      with ',len(days),len(bpds),' rates')
      ##########################
      # Loop over realizations #
      # to model stresses      #
      ##########################
      for iReal in range(self.nReal):
        (pressures[iwc,iReal],stresses[iwc,:,iReal])=self.poroelasticScenario(bpds,days,eqDay,dx,dy,dist,az,dip,self.ntBin,iReal,verbose)
      if verbose>1:
        print( '     maximum pressure: ',np.nanmax(pressures[iwc,:]),', stresses: ',np.nanmax(stresses[iwc,0,:]),np.nanmax(stresses[iwc,1,:]),np.nanmax(stresses[iwc,2,:]),np.nanmax(stresses[iwc,3,:]))
        print('      NaNs - pressure: ',sum(np.isnan(pressures[iwc,:])),', stresses: ',sum(np.isnan(stresses[iwc,0,:])),sum(np.isnan(stresses[iwc,1,:])),sum(np.isnan(stresses[iwc,2,:])),sum(np.isnan(stresses[iwc,3,:])))
    ##################################
    # Form dataframe of realizations #
    ##################################
    scenarios=pd.DataFrame(columns=['EventID','EventLatitude','EventLongitude','API','Name','ID','Latitude','Longitude','NumWells','Stresses','TotalStress','Percentages','Realization'])
    ###############################
    # Loop over realizations      #
    # for disaggregation to wells #
    ###############################
    for iReal in range(self.nReal):
      scenarioPE=pd.DataFrame(columns=['EventID','EventLatitude','EventLongitude','API','Name','ID','Latitude','Longitude','NumWells','Stresses','TotalStress','Percentages','Realization'])
      #############################################
      # Disaggregation of stresses is complicated #
      # It needs a separate function              #
      #############################################
      (percentages[:,iReal],poroStress[:,iReal],totalStresses[:,iReal],thetaVec[:,iReal])=self.poroAttr(pressures[:,iReal],stresses[:,:,iReal],az,dip,iReal,verbose)
      ##################################################
      # Number of wells with a meaningful contribution #
      ##################################################
      nwCnz=sum(percentages[:,iReal]>0.01)
      ############################
      # Set columns of dataframe #
      ############################
      scenarioPE['Stresses']=poroStress[:,iReal]
      scenarioPE['Percentages']=percentages[:,iReal]
      scenarioPE['TotalStress']=totalStresses[:,iReal]
      scenarioPE['EventID']=eq['EventID']
      scenarioPE['EventLatitude']=eq['Latitude']
      scenarioPE['EventLongitude']=eq['Longitude']
      scenarioPE['API']=consideredWells['APINumber']
      scenarioPE['Name']=consideredWells['WellName']
      scenarioPE['ID']=consideredWells['ID']
      scenarioPE['Latitude']=consideredWells['SurfaceHoleLatitude']
      scenarioPE['Longitude']=consideredWells['SurfaceHoleLongitude']
      scenarioPE['NumWells']=nwCnz
      scenarioPE['Theta']=thetaVec[:,iReal]
      ##########################
      # Add realization number #
      ##########################
      scenarioPE['Realization']=iReal
      #######################
      # Append to scenarios #
      #######################
      scenarios=pd.concat([scenarios,scenarioPE],ignore_index=True)
    ####################
    # Return scenarios #
    ####################
    return scenarios
  
  def poroelasticScenario(self,bpds,days,eqDay,dx,dy,r,azimuth,dip,ntBin,iReal,verbose=0):
    """
    ########################
    # Poroelastic modeling #
    ########################
    # Inputs: ###########################################################
    #            bpds: list of injection rates            (barrels/day) #
    #            days: list of injection days     (day, fixed sampling) #
    #           eqDay: day number of earthquake                   (day)  #
    #              dx: x-distance to earthquake                (meters) #
    #              dy: y-distance to earthquake                (meters) #
    #               r: distance to earthquake                  (meters) #
    #         azimuth: strike of fault plane  (degrees, 0=N, clockwise) #
    #             dip: dip of fault plane (degrees, right-hand, 0=flat) #
    #           ntBin: time oversampling factor              (unitless) #
    #           iReal: realization number                    (unitless) #
    #         verbose: level of verbosity                       (0,1,2) #
    #####################################################################
    # Outputs: ###############################################
    #              dP: change in pressure (PSI)              #
    #              dS: change in stress (PSI, [xx,yy,xy,zz]) #
    # Assumptions: ###########################################
    #     Run inside runPoroelasticScenarios #
    ##########################################
    """
    # Initialize dP,dS #
    ####################
    dP=0.; dS=0.
    ##########################################
    # Array of time in seconds to earthquake #
    ##########################################
    priorDays=days[days<eqDay].to_list()
    priorSec=[24*60*60*(eqDay-day) for day in priorDays]
    
    priorBPD=bpds[days<eqDay].to_list()
    
    ##############################
    # Count number of time steps #
    ##############################
    nd=np.count_nonzero(days<eqDay)
    #########################################################
    # If the earthquake was before any injection, return 0s #
    #########################################################
    if nd==0: return (0.,[0.,0.,0.,0.])
    # Convert bpds and days to m3/s/m and s
    #sec=24*60*60*(priorDays) # Seconds from injection time relative to epoch
    sec=[24*60*60*day for day in priorDays]
    eqSec=eqDay*24*60*60
    minSec=min(sec)
    ########################################
    # Original time sampling increment     #
    # Here we assume regular time sampling #
    ########################################
    dt=sec[1]-sec[0]
    ##############################
    # Interpolated time sampling #
    ##############################
    dtp=dt/ntBin
    ##############################
    # Get rates in kg per second #
    # Include density vector     #
    ##############################
    q=[self.rhoVec[iReal] * 1.84013e-6 * bpd for bpd in priorBPD] 
    #########################################
    # Go from mass/time to mass/time/length #
    # Q'0 in Equation 2.26   kg/s/m         #
    #########################################
    qp=q/self.hMVec[iReal]
    # Check inputs
    if verbose>1: print( 'poroelasticScenario: parameter check: dt, dtp, rhoVec, maxQP',dt,dtp,self.rhoVec[iReal],max(qp))
    #######################################
    # Interpolate injection data by ntBin #
    # Should I use a different method?    #
    # interp1d sets up the interpolation  #
    #######################################
    qinterp=si.interp1d(sec,qp,kind='previous',fill_value="extrapolate")
    ########################################
    # tp is the desired output time series #
    ########################################
    #tp=np.arange(0.,eqSec,dtp)
    tp=np.arange(minSec,eqSec,dtp)

    # Check outputs
    if verbose>1: print(' poroelasticScenario: tp min,max,length',min(tp),max(tp),len(tp))
    ########################################################
    # call interpolation function to get the output series #
    ########################################################
    qptp=qinterp(tp)
    if verbose>1: print(' poroelasticScenario: qptp min,max,length',min(qptp),max(qptp),len(qptp))
    ####################################
    # Compute constants: I should have #
    # computed these in the init phase #
    ##################################################
    # Leading multiplicative factor in equation 2.33 #
    # Used for the coupled pore pressure solution    #
    ##################################################
    C1= 1./(4.*np.pi*self.rhoVec[iReal]*self.kappaVec[iReal])
    ###################################################
    # Leading multiplicative factor in equation 2.34  #
    # Used for the coupled stress solution - this is  #
    # where Rudnicki missed a p0 in his equation 45   #
    ###################################################
    C2= self.muVec[iReal]*(self.lamda_uVec[iReal]-self.lamdaVec[iReal])/(np.pi*self.alphaVec[iReal]*self.rhoVec[iReal]*(self.lamda_uVec[iReal]+2*self.muVec[iReal])*r**2)
    #
    # Check C1 C2
    
    if verbose>1: print( 'poroelasticScenario: C1 C2 check:',C1,C2)
    ####################################
    # Initialize pressure and stresses #
    ####################################
    pw=np.zeros([nd,])
    Sw_xx=np.zeros([nd,])
    Sw_yy=np.zeros([nd,])
    Sw_xy=np.zeros([nd,])
    Sw_xz=np.zeros([nd,])
    Sw_yz=np.zeros([nd,])
    ####################################
    # Sanity check on input parameters #
    ####################################
    if (self.diffPEVec[iReal]<0): print("poroelasticScenario: diffPE negative!",self.diffPEVec[iReal])
    #################################
    # koxi defined in equation 2.35 #
    #################################
    koxi=r/np.sqrt(self.diffPEVec[iReal]*(eqSec-tp))
    #####################################################################
    # G1 is the arugment of the exponentials in equations 2.33 and 2.34 #
    #####################################################################
    G1 =np.exp(-0.25*koxi*koxi)
    #############################################################
    # G2 is the multiplier for the second term in equation 2.34 #
    #############################################################
    G2 =0.5*koxi*koxi
    ##################################################
    # F1 has the terms to be summed in equation 2.33 #
    ##################################################
    F1=qptp/(eqSec-tp)*G1*dtp

    # Other terms
    if verbose>1: print( 'poroelasticScenario: koxi,G1,G2,F1,eqSec,tp min/max, dtp:',max(koxi),max(G1),max(G2),max(F1),eqSec,min(tp),max(tp),dtp)
    ##########################################################
    # F2 has the terms inside the summation in equation 2.34 #
    ##########################################################
    F2_xx=qptp*((1-2*dx*dx/(r*r))*(1-G1)-(1-(dx*dx)/(r*r))*G2*G1)*dtp 
    F2_yy=qptp*((1-2*dy*dy/(r*r))*(1-G1)-(1-(dy*dy)/(r*r))*G2*G1)*dtp
    F2_xy=qptp*((0-2*dx*dy/(r*r))*(1-G1)-(0-(dx*dy)/(r*r))*G2*G1)*dtp
    #####################################
    # Summing results for 2.33 and 2.34 #
    #####################################
    F1_int=np.sum(F1)
    F2_xx_int=np.sum(F2_xx)
    F2_yy_int=np.sum(F2_yy)
    F2_xy_int=np.sum(F2_xy)  
    #################################
    # Leading scaling term for 2.33 #
    #################################
    pw= C1*F1_int
    ##########################
    # Convert from Pa to PSI #
    ##########################
    dP=pw/6894.76 
    ##########################################
    # Scaling for equation 2.34              #
    # Flip xx and yy to compression positive #
    # xz and yz are 0 under plain strain     #
    ##########################################
    Sw_xx=C2*F2_xx_int *(-1)
    Sw_yy=C2*F2_yy_int *(-1)
    Sw_xy=C2*F2_xy_int
    ###########################
    # Need a reference for zz #
    ###########################
    Sw_zz=self.nuVec[iReal]*(Sw_xx+Sw_yy)
    ###################################
    # Convert stresses from Pa to PSI #
    ###################################
    dS=[Sw_xx/6894.76,Sw_yy/6894.76,Sw_xy/6894.76,Sw_zz/6894.76]
    return(dP, dS)
    
  def poroAttr(self,pressures,stresses,azimuth,dip,iReal,verbose):
    """
    ##############################
    # Poroelastic disaggregation #
    ##############################
    # Inputs: ###########################################################
    #       pressures: array of per-well pressures[nw]            (PSI) #
    #        stresses: array of per-well stresses [nw,4]          (PSI) #
    #         azimuth: strike of fault plane  (degrees, 0=N, clockwise) #
    #             dip: dip of fault plane (degrees, right-hand, 0=flat) #
    #           iReal: realization number                     (integer) #
    #         verbose: debugging verbosity                      (0,1,2) #
    #####################################################################
    # Outputs: #######################################################
    #            perc: per-well Coulomb stress change [nw] (percent) #
    #             CFF: per-well Coulomb stress change [nw]     (PSI) #
    #          CFFsum: total change in Coulomb stress          (PSI) #
    #        thetaVec: Slip direction CCW from TF1         (degrees) #
    # Assumptions: ###################################################
    #     Run inside runPoroelasticScenarios #
    ##########################################
    """
    # First convert azimuth, dip to radians #
    #########################################
    azr=np.radians(azimuth)
    dipr=np.radians(dip)
    
    ########################################################
    # Number of wells with contributing pressures/stresses #
    ########################################################
    nw=len(pressures)
    ##################################
    # Initialize CFF and percentages #
    ##################################
    CFF=np.zeros([nw,1])
    perc=np.zeros([nw,1])
    CFFsum=np.zeros([nw,1])
    thetaVec=np.zeros([nw,1])
    ##########################################
    # Sum stress/pressure changes over wells #
    # Convert from PSI back to Pa            #
    ##########################################
    ssv=np.sum(stresses,axis=0)*6894.76
    pp=np.sum(pressures)* 6894.76
    ################################################
    # Form effective stress tensor change for this #
    # earthquake - the blue terms in equation 4.2  #
    ################################################
    ss=formEffectiveStressTensor(ssv,pp)
    ##########################################
    # Compute unit vectors nf, tr1 and tr2   #
    # nf - direction normal to fault         #
    # tr1/2 - orthogonal directions on fault #
    # phi=azimuth delta=dip                  #
    ##########################################
    nf=[np.sin(dipr)*np.cos(azr), -np.sin(dipr)*np.sin(azr), np.cos(dipr)]
    tf1=[np.sin(azr), np.cos(azr), 0.]
    tf2=[-np.cos(dipr)*np.cos(azr), np.cos(dipr)*np.sin(azr), np.sin(dipr)]
    #####################################################################
    # Project stresses onto fault plane: Equations 4.14 to 4.16         #
    # Input - full effective stress tensor (delta), unit vectors        #
    # Output - traction, normal, and share stresses for combined stress #
    #####################################################################
    (tr,sn,st1,st2) = projectStress(ss,nf,tf1,tf2)
    ############################################################################
    # theta = slip tendency direction in degrees (CCW from tf1), equation 4.20 #
    ############################################################################
    # Need to switch to arctan2 - st1 could be zero
    #theta=np.degrees(np.arctan(st2/st1))
    theta=np.degrees(np.arctan2(st2,st1))
    thetaVec[:,0]=theta
    for iw in range(nw):
      ##############################################################################
      # Form effective stress tensor for each well contributing to this earthquake #
      ##############################################################################
      ssw=formEffectiveStressTensor(stresses[iw,:],pressures[iw])
      #################################################
      # Project effective stress tensor for each well #
      #################################################
      (trw,snw,st1w,st2w)=projectStress(ssw,nf,tf1,tf2)
      ###################################################
      # Get angle on fault for this well: Equation 4.21 #
      ###################################################
      # To-do - fix the case with no shear stress (NaN) #
      ###################################################
      #theta_w=np.degrees(np.arctan(st2w/st1w))
      theta_w=np.degrees(np.arctan2(st2w,st1w))
      #####################################
      # beta for each well: Equation 4.22 #
      # If cos(beta)<0 - stabilizes fault #
      # What about phase jumps here?
      #####################################
      beta=theta_w-theta
      #####################################
      # fault-normal part of CFF for well #
      # Single place where muF gets used  #
      # Term 4.25 in Equation 4.26        #
      #####################################
      CFF_n= -self.muFVec[iReal]*snw
      ##########################################
      # fault-tangential part of CFF for well  #
      # Term 4.24 (without cosine) in Eqn 4.26 # 
      # equivalent to sqrt(st1w^2+st2w^2)      #
      ##########################################
      CFF_t=np.sqrt(np.linalg.norm(trw)**2-snw**2)
      ###################################################
      # "relevant" fault-tangential part of CFF by well # 
      # CFF along the final slip tendency direction     #
      ###################################################
      if np.isnan(beta)==False:
        ################################################
        # Project shear stress on final slip direction #
        # Cosine part of first term in Equation 4.26   #
        ################################################
        CFF_t=CFF_t*np.cos(np.radians(beta))
      else:
        ########################################
        # Special case: if there is no shear   #
        # traction there is nothing to project #
        ########################################
        CFF_t=0.  
      ######################################
      # Combine two terms of equation 4.26 #
      ######################################
      CFF[iw,0]=CFF_t+CFF_n
 
      # Contribution 
    #% total CFF by all wells at eth source faults on kth time step 
    ########################################
    # Create array of all total CFF values #
    ########################################
    CFFsum=np.sum(CFF)*np.ones([nw,1])
    #############################################
    # Compute percentage conributions to stress #
    #############################################
    for iw in range(nw):
      perc[iw,0]=100.*CFF[iw,0]/CFFsum[iw,0]
    ##############################################################
    # Return lists of percentages, CFFs, total CFF for all wells #
    ##############################################################
    return (perc[:,0],CFF[:,0],CFFsum[:,0],thetaVec[:,0])


  def pressureScenariosToDF(self,eq,consideredWells,dPAtEQ,totalPressureAtEQ,percentages):
    """
    # Create dataframe for pressure scenarios
    Inputs: pressureScenariosToDF(eq,consideredWells,dPatEQ,totalPressureAtEQ,percentages)
          eq:                 Dataframe of earthquake
                              with EventID, Latitude, and Longitude columns
          consideredWells:    Dataframe of wells from findWells
                              with APINumber, WellName, ID, SurfaceHoleLatitude, and SurfaceHoleLongitude
          dPAtEQ:             Numpy array of per-well pressure contributions at earthquake epicenter/Time
                              size(consideredWells.shape()[0], self.nReal)
          totalPressureAtEQ:  Numpy array of total pressures at earthquake epicenter/time
                              size(self.nReal)
          percentages:        Numpy array of per-well relative contributions at earthquake epicenter/Time
                              size(consideredWells.shape()[0], self.nReal)
    Returns:
          scenarioDF:         Dataframe with consideredWells.shape()[0] x self.nReal rows
    """
    allScenariosDF=pd.DataFrame(columns=['EventID','EventLatitude','EventLongitude','ID','Name','API','Latitude','Longitude','NumWells','Pressures','TotalPressure','Percentages','Realization'])
    ##################################################
    # Number of wells with a meaningful contribution #
    ##################################################
    nw=consideredWells.shape[0]
    #nwCnz=np.sum((percentages>0.01),axis=0)
    print('pressureScenariosToDF: number of wells: ',nw)
    print('pressureScenariosToDF: number of realizations:',self.nReal)
    print('pressureScenariosToDF: dPAtEQ:',dPAtEQ.shape,len(dPAtEQ.flatten()))
    print('pressureScenariosToDF: totalPressureAtEQ:',totalPressureAtEQ.shape,len(totalPressureAtEQ.flatten()))
    
    # Ordering of dataframe is in fast to slow:
    # Well, realization, event

    ############################
    # Set columns of dataframe #
    ############################
    #scenarioDF['Pressures']=dPAtEQ.flatten()
    #scenarioDF['TotalPressure']=np.repeat(totalPressureAtEQ.flatten(),nw)
    #scenarioDF['Percentages']=percentages.flatten()
    #scenarioDF['EventID']=eq['EventID']
    #scenarioDF['EventLatitude']=eq['Latitude']
    #scenarioDF['EventLongitude']=eq['Longitude']
    #apis=np.tile(consideredWells['APINumber'].to_numpy(),self.nReal)
    #print('pressureScenariosToDF: ',len(apis),scenarioDF.shape[0])
    #scenarioDF['API']=apis
    ##scenarioDF['Name']=pd.concat([consideredWells['WellName'].squeeze()]*self.nReal,axis=0).reset_index()
    ##scenarioDF['Name']=np.tile(consideredWells['WellName'].to_numpy(),self.nReal)
    #scenarioDF['ID']=np.tile(consideredWells['ID'].to_numpy(),self.nReal)
    #scenarioDF['Latitude']=np.tile(consideredWells['SurfaceHoleLatitude'].to_numpy(),self.nReal)
    #scenarioDF['Longitude']=np.tile(consideredWells['SurfaceHoleLongitude'].to_numpy(),self.nReal)
    #scenarioDF['NumWells']=np.repeat(nwCnz,nw)
    ##########################
    # Add realization number #
    ##########################
    #scenarioDF['Realization']=np.repeat(np.linspace(0,self.nReal-1,self.nReal),nw)

    
    ##################################
    # Form dataframe of realizations #
    ##################################
    totalPressures=np.zeros([nw,self.nReal])
    #percentages=np.zeros([nw,self.nReal])
    ###############################
    # Loop over realizations      #
    # for disaggregation to wells #
    ###############################
    for iReal in range(self.nReal):
      scenarioDF=pd.DataFrame(columns=['EventID','EventLatitude','EventLongitude','ID','Name','API','Latitude','Longitude','NumWells','Pressures','TotalPressure','Percentages','Realization'])
      ######################################
      # Sum pressures for this realization #
      # and disaggregate - equation 4.9    #
      ######################################
      totalPressures[:,iReal]=np.sum(dPAtEQ[:,iReal])
      #percentages[:,iReal]=100.*dPAtEQ[:,iReal]/totalPressures[:,iReal]
      ##################################################
      # Number of wells with a meaningful contribution #
      ##################################################
      nwCnz=sum(percentages[:,iReal]>0.01)
      ############################
      # Set columns of dataframe #
      ############################
      scenarioDF['Pressures']=dPAtEQ[:,iReal]
      scenarioDF['TotalPressure']=totalPressures[:,iReal]
      scenarioDF['Percentages']=percentages[:,iReal]
      scenarioDF['EventID']=eq['EventID']
      scenarioDF['EventLatitude']=eq['Latitude']
      scenarioDF['EventLongitude']=eq['Longitude']
      scenarioDF['API']=consideredWells['APINumber']
      scenarioDF['Name']=consideredWells['WellName']
      scenarioDF['ID']=consideredWells['ID']
      scenarioDF['Latitude']=consideredWells['SurfaceHoleLatitude']
      scenarioDF['Longitude']=consideredWells['SurfaceHoleLongitude']
      scenarioDF['NumWells']=nwCnz
      ##########################
      # Add realization number #
      ##########################
      scenarioDF['Realization']=iReal
      #######################
      # Append to scenarios #
      #######################
      allScenariosDF=pd.concat([allScenariosDF,scenarioDF],ignore_index=True)
    return allScenariosDF

# Generic subroutines not inside the class

def prepInj(consideredWells,injDF,dt,dxdyIn=None,eqDay=None,endDate=None):
  """
  prepInj(consideredWells,injDF,endDate=None)
  Produce numpy arrays of injection rates from 
  well and injection dataframes. Missing data 
  will be zeroes, assumes this data is regularized
  in time to a common time increment!!
  Inputs:
    consideredWells - well dataframe from findWells
    injDF - injection dataframe from findWells
    dt    - time interval of injection data in days
    eqDay - earthquake day, used for timing, optional
    endDate - desired otuput
    dxdyIn - grid values relative to EQ location - nx x ny, 2
  Outputs:
    wellIDs - well numbers in order of bpdArray
    nwC - number of wells
    dayArray - vector of day numbers
    nt, ot - sampling of bpdArray in days
    bpdArray(nw,nt) - rates in barrels per day
    secArray(nt)
    dx (nw)
    dy (nw)
    wellDistances (nw)
    ieq, f - index of earthquake, interpolation weight (if eqDay provided)
    dxdyArray(nw x nx x ny, 2) if dxdyIn provided

  Rates in BPD, distances in m, time in days
  Since we are taking differences, we prepend by one time sample with a zero
  """
  nwC=consideredWells.shape[0]
  # Form numpy arrays of well numbers, distances
  wellIDs=consideredWells['ID'].to_numpy()
  wellDistances=1000.*consideredWells['Distances'].to_numpy()
  dx=1000.*consideredWells['DXs'].to_numpy()
  dy=1000.*consideredWells['DYs'].to_numpy()
  # Create Grid Relative to Earthquake here
  if dxdyIn is not None:
    dxdyArray=np.zeros([nwC, dxdyIn.shape[0],dxdyIn.shape[1],2])
    for iw in range(nwC):
      print('dx, dy ',dx[iw],dy[iw])
      dxdyArray[iw,:,:,0]=dx[iw]-1000.*dxdyIn[:,:,0]
      dxdyArray[iw,:,:,1]=dy[iw]-1000.*dxdyIn[:,:,1]
    dxdyArray=dxdyArray
  # By default, make everything relative to the last injection date #
  minT=min(injDF['Days'])
  # Need to prepend by a zero since we will take a difference
  ot=minT-dt
  # If endDate is provided make it relative to that 
  if endDate is not None:
    maxT=endDate
  elif eqDay is not None:
    maxT=max(int((eqDay-ot)/dt)*dt,max(injDF['Days']))+dt
  else:
    maxT=max(injDF['Days'])+dt
  nt=int((maxT-ot)/dt)+1
  #nt=nt1+1
  # Form array (nWells x ntOut of input injection)
  bpdArray=np.zeros([nwC,nt])
  # dayArray is the duration in days from each step to the end of the simulation
  dayArray=np.linspace(start=ot,stop=maxT,num=nt,endpoint=False)
  # secArray is dayArray in seconds
  secArray=24*60*60*dayArray
  # Loop over wells
  for iw in range(nwC):
    # find BPD and Days values that match this well ID, make a list #
    bpds=injDF['BPD'][injDF['ID']==consideredWells['ID'][iw]].tolist()
    days=injDF['Days'][injDF['ID']==consideredWells['ID'][iw]].tolist()
    # Check if we have any injection for this well
    # Should I put a warning here if we have no injection data? #
    if len(days)>0:
      for id in range(len(days)):
        # Get index of day value - this should be exact #
        it=int((days[id]-ot)/dt)
        bpdArray[iw,it]=bpds[id]
  # Evaluate dP at time of earthquake if eqDay provided
  if eqDay is not None:
    # Get point on time axis in terms of indicies of the output (ot, not ot0)
    feq=(eqDay-ot)/dt
    # Get the first point
    ieq=int(feq)
    # Get a linear interpolation weight
    f=feq-ieq
    return (wellIDs,nwC,dayArray,nt,ot,bpdArray,secArray,dx,dy,wellDistances,ieq,feq)
  elif dxdyIn is not None:
    return (wellIDs,nwC,dayArray,nt,ot,bpdArray,secArray,dx,dy,wellDistances,dxdyArray)
  else:
    return (wellIDs,nwC,dayArray,nt,ot,bpdArray,secArray,dx,dy,wellDistances)

def calcPPVals(kMD,hFt,alphav,beta,phi,rho,g,nta):
  """
  ############################################################
  # Convert inputs to SI units and generate other parameters #
  ############################################################
  # Inputs: #######################################
  #   kMD:    permeability         (millidarcies) #
  #   hFt:    thickness                    (feet) #
  #   alphav: vertical compressibility     (1/Pa) #
  #   beta:   fluid compressibility        (1/Pa) #
  #   phi:    porosity                  (percent) #
  #   rho:    fluid density              (kg/m^3) #
  #   g:      gravitational acceleartion  (m/s^2) #
  #   nta:    fluid viscosity    (Pascal seconds) #
  #################################################
  # Outputs: ###################################
  #   phiFrac: porosity           (fractional) #
  #   hM:      thickness              (meters) #
  #   kapM2:   permeability in           (m^2) #
  #   S:       storativity          (unitless) #
  #   K:       conductivity      (kg / Pa s^3) #
  #   T:       transmissivity        (m^2 / s) #
  #   diffPP:  diffusivity           (m^2 / s) #
  #   C:       lumped compressibility   (1/Pa) #
  ##############################################
  """
  # Unit conversions #
  ####################################
  # Convert from Millidarcies to m^2 #
  ####################################
  kapM2 = kMD*(1e-3)*(9.9e-13)
  ##################
  # Feet to meters #
  ##################
  hM=hFt*0.3048
  #######################
  # Percent to fraction #
  #######################
  phiFrac=phi*0.01
  ###########################
  # Intermediate parameters #
  #######################################
  # Now compute intermediate parameters #
  #######################################
  # Diffusivity (pore pressure) #
  # Equation 1.29, 3.1          #               
  ###############################
  diffPP=kapM2/(nta*(alphav+(beta*phiFrac)))
  ##########################
  # Lumped compressibility #
  ##########################
  C=alphav+phiFrac*beta
  ################
  # Storativity  #
  # Equation 1.4 #
  ################
  S=rho*g*hM*C
  ####################################
  # Saturated hydraulic conductivity #
  ####################################
  K = kapM2*rho*g/nta
  ##################
  # Transmissivity #
  # Equation 1.5   #
  ##################
  T = K*hM
  return (phiFrac,hM,kapM2,S,K,T,diffPP,C)

def calcPPAnisoVals(kMDTensor,hM,alphav,beta,phi,rho,g,nta):
  """
  ##########################################################################
  # Convert tensor inputs to SI units and generate other tensor parameters #
  ##########################################################################
  # Inputs: #######################################
  #   kMD:    permeability     (2x2,millidarcies) #
  #   hM:     thickness                  (meters) #
  #   alphav: vertical compressibility     (1/Pa) #
  #   beta:   fluid compressibility        (1/Pa) #
  #   phi:    porosity                  (percent) #
  #   rho:    fluid density              (kg/m^3) #
  #   g:      gravitational acceleartion  (m/s^2) #
  #   nta:    fluid viscosity    (Pascal seconds) #
  #################################################
  # Outputs: ###################################
  #   kapM2:   permeability in       (2x2,m^2) #
  #   K:       conductivity      (kg / Pa s^3) #
  #   T:       transmissivity    (2x2,m^2 / s) #
  #   diffPP:  diffusivity       (2x2,m^2 / s) #
  ##############################################
  """
  # Unit conversions #
  ####################################
  # Convert from Millidarcies to m^2 #
  ####################################
  kapM2 = kMDTensor*(1e-3)*(9.9e-13)
  #######################
  # Percent to fraction #
  #######################
  phiFrac=phi*0.01
  ###########################
  # Intermediate parameters #
  #######################################
  # Now compute intermediate parameters #
  #######################################
  # Diffusivity (pore pressure) #
  # Equation 1.29, 3.1          #               
  ###############################
  diffPP=kapM2/(nta*(alphav+(beta*phiFrac)))
  ####################################
  # Saturated hydraulic conductivity #
  ####################################
  K = kapM2*rho*g/nta
  ##################
  # Transmissivity #
  # Equation 1.5   #
  ##################
  T = K*hM
  # 
  TBar=np.sqrt( T[0,0]*T[1,1] - T[0,1]*T[1,0])
  return (kapM2,K,T,diffPP,TBar)


def matchPE2PP(mu,nu,alpha,C):
  """
  ##############################################################
  # Match poroelastic diffusivity to pore pressure diffusivity #
  ##############################################################
  # Inputs: ############################################
  #     mu : Rock shear modulus              (Pascals) #
  #     nu : Rock drained Poisson's ratio   (unitless) #
  #  alpha : Biot's coefficient             (unitless) #
  #      C : Lumped compressibility             (1/Pa) #
  ######################################################
  # Outputs: ####################################
  #   lamda:   Drained Lame's parameter    (Pa) #
  #   lamda_u: Undrained Lame's parameter  (Pa) #
  ###############################################
  """
  lamda=2.*mu*nu/(1.-2.*nu)
  ################
  # Equation 3.5 #
  ################
  A=lamda+2*mu
  ################
  # Equation 3.6 #
  ################
  lamda_u=(2.*alpha*alpha*mu + C*A*lamda)/(C*A - alpha*alpha)
  return lamda,lamda_u

def calcPEVals(mu,nu,nu_u,alpha,kapM2,nta):
  """
  ################################################
  # Calculate alternative poroelastic parameters #
  ################################################
  # Inputs: ############################################
  #     mu : Rock shear modulus              (Pascals) #
  #     nu : Rock drained Poisson's ratio   (unitless) #
  #   nu_u : Rock undrained Poisson's ratio (unitless) #
  #  alpha : Biot's coefficient             (unitless) #
  #  kapM2 : Permeability                        (m^2) #
  #    nta : Fluid viscosity                    (Pa s) #
  ######################################################
  # Outputs: ######################################
  #   lamda:   Drained Lame's parameter      (Pa) #
  #   lamda_u: Undrained Lame's parameter    (Pa) #
  #   B:       Skempton's coefficient  (unitless) #
  #   diffPE:  Poroelastic diffusivity    (m^2/s) #
  #   kappa:   Hydraulic conductivity (m^2/(Pa s) #
  #################################################
  """
  # Compute drained Lame's parameter/constant #
  # from Poisson's Ratio and Shear Modulus    #
  #############################################
  lamda=2.*mu*nu/(1.-2.*nu)
  ###############################################
  # Compute UNdrained Lame's parameter/constant #
  # from Poisson's Ratio and Shear Modulus      #
  ###############################################
  lamda_u=2.*mu*nu_u/(1.-2.*nu_u)
  #########################################
  # Skempton's coefficient                #
  # 0.75 for Berea ss (Hart & Wang, 1995) #
  #########################################
  B=3.*(nu_u-nu)/(alpha*(1.+nu_u)*(1.-2.*nu)) 
  ###########################
  # Poroelastic diffusivity #
  # Equation 3.3            #
  ###########################
  diffPE=(kapM2)*(lamda_u-lamda)*(lamda+2.*mu)/(nta*alpha*alpha*(lamda_u+2.*mu))
  kappa=kapM2/nta
  return (lamda,lamda_u,B,diffPE,kappa)

def getDates(inDF,epoch,dayName='Days',default=99999999,verbose=0):
  """
  ############################################################
  # Creates a column of integer day numbers from input dates #
  ############################################################
  # Inputs                                      #
  #    inDF - Dataframe with Days column        #
  #    epoch - datetime value of starting date  #
  # Outputs                                     #
  #    outDF - Dataframe with added Date column #
  ###############################################
  """
  outName=dayName.replace('Days','Date')
  outDF=inDF.copy()
  dateList=[]
  if verbose>0: print(' getDates: ',len(inDF[dayName]))
  ##################
  # Loop over rows #
  ##################
  for index,row in inDF.iterrows():
    #######################
    # Access Days column #
    #######################
    day=row[dayName]
    if pd.isna(day):
      dateList.append(epoch+ pd.Timedelta(default,unit='day'))
    else:
      try:
        dateList.append(epoch+pd.Timedelta(day,unit='day'))
      except:
        print('getDates error ',day,pd.Timedelta(day),epoch)
  #print(' getDates - dateList ', dateList)
  outDF[outName]=dateList
  return outDF

def logSpace(centerVal,logUncertainty,clip=None,verbose=0):
  """
  ################################################
  # logSpace(centerVal,logUncertainty,clip=None,verbose=0) #
  ##########################################################
  # Set linear uncertainty bounds given logarithmic ranges #
  ##########################################################
  # Inputs: ###################################################################
  #        centerVal: original central deterministic value    (unknown units) #
  #   logUncertainty: log10 uncertainty - how many orders                     #
  #                   of magnitude could we be off by? (log10(unknown units)) #
  #             clip: If you want a minimum value, set it here
  #############################################################################
  """
  upperBound=np.power(10.,(np.log10(centerVal)+logUncertainty))
  lowerBound=np.power(10.,(np.log10(centerVal)-logUncertainty))
  if clip!=None: lowerBound=min(lowerBound,clip)
  uncertainty=(upperBound-lowerBound)/2.
  newCenterVal=lowerBound+uncertainty
  if verbose>0: print("  logSpace: lower bound: ",lowerBound,", upper bound: ",upperBound)
  return newCenterVal,uncertainty

def haversineXY(lat1,lat2,lon1,lon2):
  """
  ##################################################
  # Estimate X / Y distances between Lat/Lon pairs #
  # This runs Haversite twice using midpoints      #
  # Not exact, but close enough for us             #
  ##################################################
  # Inputs: ##########################################
  #   lat1:  latitude of 1st point (decimal degrees) #
  #   lat2:  latitude of 2nd point (decimal degrees) #
  #   lon1: longitude of 1st point (decimal degrees) #
  #   lon2: longitude of 2nd point (decimal degrees) #
  ####################################################
  # Outputs: ###################################
  #   dx:  approximate x-distance (kilometers) #
  #   dy:  approximate y-distance (kilometers) #
  ##############################################
  # Assumptions: #####################
  #   Sea level, earth is a sphere   #
  #   Probably breaks near the poles #
  """
  ###############################################################
  # Run Haversine once with latitudes, taking average longitude #
  ###############################################################
  dy=haversine(lat1,lat2,0.5*(lon1+lon2),0.5*(lon1+lon2))
  ######################################################
  # Run again with longitudes, taking average latitude #
  ######################################################
  dx=haversine(0.5*(lat1+lat2),0.5*(lat1+lat2),lon1,lon2)
  #######################################################
  # Sign conventions, negative is Southwest from 1 to 2 #
  #######################################################
  if lat1<lat2: dy=-dy
  if lon1<lon2: dx=-dx
  return [dx,dy]
  
def haversine(lat1,lat2,lon1,lon2):
  """
  ###################################################
  # Great-circle distance between two lat/lon pairs #
  ###################################################
  # Inputs: ##########################################
  #   lat1:  latitude of 1st point (decimal degrees) #
  #   lat2:  latitude of 2nd point (decimal degrees) #
  #   lon1: longitude of 1st point (decimal degrees) #
  #   lon2: longitude of 2nd point (decimal degrees) #
  ####################################################
  # Outputs: ####################
  #   d:  distance (kilometers) #
  ###############################
  # Assumptions: #####################
  #   Sea level, earth is a sphere   #
  #   Probably breaks near the poles #
  ##########################################
  """
  # Convert inputs from degrees to radians #
  ##########################################

  rlat1=math.radians(lat1)
  rlat2=math.radians(lat2)
  rlon1=math.radians(lon1)
  rlon2=math.radians(lon2)
  ###############
  # Differences #
  ###############
  dlon=rlon2-rlon1
  dlat=rlat2-rlat1
  a=math.sin(dlat/2)**2 + math.cos(rlat1)*math.cos(rlat2)*math.sin(dlon/2)**2
  try:
    c=2*math.atan2(math.sqrt(a),math.sqrt(1.-a))
  except:
    print('haversine error: ',a,dlat,dlon,rlat1,rlat2)
  ######################################
  # Scale by radius of the earth in km #
  ######################################
  d=6373.0*c
  return d

def formEffectiveStressTensor(stresses,pressure):
  """
  #####################################################################
  # Forms 3x3 effective stress tensor from pressure and four stresses #
  #####################################################################
  # Inputs: ##############################################
  #   stresses: Stress tensor [xx,yy,xy,zz]  (PSI or Pa) #
  #   pressure: Pressure scalar       (same as stresses) #
  ########################################################
  # Output: ################################################
  #   ss: Effective stress tensor [3,3] (same as stresses) #
  ##########################################################
  """
  # Initialize output tensor #
  ############################
  ss=np.zeros([3,3])
  ############################################
  # Subtract pore pressure from the diagonal #
  ############################################
  ss[0,0]=stresses[0]-pressure
  ss[0,1]=stresses[2]
  ss[1,0]=stresses[2]
  ss[1,1]=stresses[1]-pressure
  ss[2,2]=stresses[3]-pressure
  ####################################
  # Sxz and Syz are 0 - plain strain #
  ####################################
  ss[2,0:2]=0.
  ####################################
  # Szx and Szy are 0 - plain strain #
  ####################################
  ss[0:2,2]=0.
  return ss
  
def projectStress(ss,nf,tf1,tf2): 
  """
  ############################################
  # Projects stress onto fault plane vectors #
  ############################################
  # Inputs: #####################################
  #   ss:  Input stress tensor      (PSI or Pa) #
  #   nf:  Fault-normal unit vector  (unitless) #
  #   tf1: Along-fault unit vector 1 (unitless) #
  #   tf2: Along-fault unit vector 2 (unitless) #
  ###############################################
  # Outputs: ###################################
  #   tr:  Traction               (same as ss) #
  #   sn:  Normal stress          (same as ss) #
  #   st1: Shear stress along tf1 (same as ss) #
  #   st2: Shear stress along tf2 (same as ss) #
  ##############################################
  """
  tr=np.matmul(ss,nf)
  ######################
  # equations 4.14-4.16 #
  ######################
  sn=np.sum(tr*nf)
  st1=np.sum(tr*tf1)
  st2=np.sum(tr*tf2)
  return (tr,sn,st1,st2)
  
def stressInvariants(Sxx,Syy,Szz,Sxy,Sxz,Syz,mu_r):
  """
  #############################
  # Compute stress invaraints #
  #############################
  # Inputs: #############################################
  #   S??:  6 independent stress components (PSI or Pa) #
  #   mu_r: Rock friction coefficient        (unitless) #
  #######################################################
  # Outputs: ###################################
  #   mean: Mean stress          (same as S??) #
  #    dev: Deviatoric stress    (same as S??) #
  #     mc: Mohr-Coulomb? stress (same as S??) #
  ##############################################
  """  
  mean=(Sxx + Syy + Szz)/3.
  dev=np.sqrt(((Sxx-Syy)**2. + (Sxx-Szz)**2. + (Syy-Szz)**2. + 6.*(Sxy**2. + Sxz**2. + Syz**2.) )/6. )
  MC=dev - np.sin(np.arctan(mu_r))*mean
  return [mean,dev,MC]
