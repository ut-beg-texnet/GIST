# GIST

Geomechanical Injection Scenario Toolkit

Dependencies: Python (3.9.10), numpy (1.21.4), scipy (1.7.3), pandas (1.3.5)

Additional dependencies for plotting: seaborn, matplotlib, geopandas, contextily

Code is written as classes and subroutines in one .py file, with a second .py file or Jupyter notebook as a driver script

Injection processing, GIST computation, and visualization are broken up into separate steps

# Contents of compute branch:

	gistMC.py               		- GIST class built for version 3 of the B3 data
  
	gistMCLiveTarzan.py         		- driver script built around the Range Hill earthquake

	gistPlots.py		    		- collection of plotting functions to visualize results

	GIST_Stanton_11-4-23_Compute.ipnyb	- Jupyter notebook for 11/4/23 Stanton earthquake - computation

	GIST_Stanton_11-4-23Plots.ipynb		- Jupyter notebook for 11/4/23 Stanton earthquake - plotting

	GIST_TestAnisotropy.ipynb		- Jupyter notebook for testing azimuthal permeability anisotropy in gistMCLive

   	GIST_Stanton_TodayCompute.ipynb		- Jupyter notebook for a hypothetical present-day Stanton earthquake - computation

# To-do (very incomplete list)

	- Validate pore pressure modeling codes
  
	- Include Matlab prototype code
  
	- Regression tests for Python code to match "Gold" results from Matlab code

 	- Rework series of discrete steps into a workflow:
  		- QC and edit injection data after well selection
    		- Update GIST parameterization after plots
  
	- Anisotropy examples
  
	- Incorporate BEG-checked data from Bob Reedy
