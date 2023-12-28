# GIST

Geomechanical Injection Scenario Toolkit

Dependencies: Python (3.9.10), numpy (1.21.4), scipy (1.7.3), pandas (1.3.5)

Code is written as classes and subroutines in one .py file, with a second .py file as a driver script

# Contents:

Code to regenerate results from IMAGE presentation in August 2022:

		gistMC.py                   - first Monte-Carlo GIST code used for IMAGE

		gistMCExample.py            - driver script for IMAGE examples


Injection processing code:

		injectionV3.py              - basic injection data processing/merging for B3 TX and NM injection .csv files, (version 3)

		injectionV3_driver.py       - driver script to output time-regularized and filtered injection data with daily sampling

		injectionV3_driver_10day.py - same as above, but with 10-day sampling to make the code run a bit faster for events with large numbers of wells

Most recent version:

		gistMCLive.py               - GIST class built for version 3 of the B3 data
  
		gistMCLiveTarzan.py         - driver script built around the Range Hill earthquake

# To-do (very incomplete list)
  
	- Include Matlab prototype code
  
	- Regression tests for Python code to match "Gold" results from Matlab code
  
	- Anisotropy examples
  
	- Examples contain no visualization
  
	- Change driver scripts to .ipynb files to run in Jupyter notebooks
		- Include intermediate QC and visualization
  
	- Incorporate BEG-checked data from Bob Reedy
