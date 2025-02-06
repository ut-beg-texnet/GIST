# GIST

Geomechanical Injection Scenario Toolkit

Dependencies: Python (3.9.10), numpy (1.21.4), scipy (1.7.3), pandas (1.3.5)

Additional dependencies for plotting: seaborn, matplotlib, geopandas, contextily

Code is structured as classes and subroutines in one .py file, with a second .py file or Jupyter notebook as a driver script

Injection processing, GIST computation, and visualization are broken up into separate steps

# Contents of compute branch:

	gistMC.py               			- GIST class built for version 4 of the B3 data
  
	GIST_RunTemplate_texnet2024oqfb_deep.ipynb      - Jupyter notebook of all compute for one example

	GIST_RegressionTests.ipynb			- Jupyter notebook reproducing FSP results for modeling regression tests,
 							  validation of pore pressure modeling.

	gistSteps.py					- Individual steps from Jupyter notebook broken out for web tools.

	gistStepsDriver.py				- Driver script for gistSteps to run discrete steps matching web tool.

 	Geomechanics_GF.py				- Recently developed Green's functions for out-of-zone stressing.
  							  Documentation in progress.

    	gold/						- Directory of .csv files with results from Fault Slip Potential used as
     							  references for GIST_RegressionTests.
# To-do (very incomplete list)
  
	- Include Matlab prototype code
  
	- Anisotropy examples
  
	- Incorporate BEG-checked data


# Disclaimer

GIST aims to give the <i>gist</i> of a wide range of potential scenarios and aid collective decision making when responding to seismicity.

The results of GIST are entirely dependent upon the inputs provided, which may be incomplete or inaccurate.

There are other potentially plausible inducement scenarios that are not considered, including fluid migration into the basement, out-of-zone poroelastic stressing, or hydraulic fracturing.

None of the individual models produced by GIST accurately represent what happens in the subsurface and cannot be credibly used to accurately assign liability or responsibility for seismicity.

"All models are wrong, but some are useful" - George Box, 1976

