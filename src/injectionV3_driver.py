import scipy.special as sc
import numpy as np
import pandas as pd
import math
import injectionV3 as inj3

# Driver for injectionV3

# Step 1 - point to files - will replace this with REST API calls at some point
# Points to current B3 v3 datasets as of 10/31/22 - NM column names will change

TXWellFile='/data/xom/seismicity/shared/B3/TX/InjectionWell.csv'
TXMonthlyFile='/data/xom/seismicity/shared/B3/TX/MonthlyInjection.csv'
TXDailyFile='/data/xom/seismicity/shared/B3/TX/DailyInjection.csv'

TXDTempWFile='/data/xom/seismicity/bcurry/GIST/TestTXWellsDeep.csv'
TXDTempMFile='/data/xom/seismicity/bcurry/GIST/TestTXMonthlyInjDeep.csv'
TXDTempDFile='/data/xom/seismicity/bcurry/GIST/TestTXDailyInjDeep.csv'
TXSTempWFile='/data/xom/seismicity/bcurry/GIST/TestTXWellsShallow.csv'
TXSTempMFile='/data/xom/seismicity/bcurry/GIST/TestTXMonthlyInjShallow.csv'
TXSTempDFile='/data/xom/seismicity/bcurry/GIST/TestTXDailyInjShallow.csv'

NMWellFile='/data/xom/seismicity/shared/B3/NM/InjectionWell.csv'
NMMonthlyFile='/data/xom/seismicity/shared/B3/NM/MonthlyInjection.csv'
NMDailyFile='/data/xom/seismicity/shared/B3/NM/DailyInjection.csv'

NMDTempWFile='/data/xom/seismicity/bcurry/GIST/TestNMWellsDeep.csv'
NMDTempMFile='/data/xom/seismicity/bcurry/GIST/TestNMMonthlyInjDeep.csv'
NMDTempDFile='/data/xom/seismicity/bcurry/GIST/TestNMDailyInjDeep.csv'
NMSTempWFile='/data/xom/seismicity/bcurry/GIST/TestNMWellsShallow.csv'
NMSTempMFile='/data/xom/seismicity/bcurry/GIST/TestNMMonthlyInjShallow.csv'
NMSTempDFile='/data/xom/seismicity/bcurry/GIST/TestNMDailyInjShallow.csv'

DeepWellFile='/data/xom/seismicity/bcurry/GIST/testDeep.csv'
DeepInjFile='/data/xom/seismicity/bcurry/GIST/testDeepInj.csv'
DeepPrefix='/data/xom/seismicity/bcurry/GIST/testDeep'

ShallowWellFile='/data/xom/seismicity/bcurry/GIST/testShallow.csv'
ShallowInjFile='/data/xom/seismicity/bcurry/GIST/testShallowInj.csv'
ShallowPrefix='/data/xom/seismicity/bcurry/GIST/testShallow'
# First deal with Deep
# Texas
TXDInj=inj3.injTX(TXWellFile,'Deep',7000.,TXDTempWFile)
TXDInj.addMonthly(TXMonthlyFile,1000000,TXDTempMFile)
TXDInj.addDaily(TXDailyFile,100000,TXDTempDFile)

# New Mexico
NMDInj=inj3.injNM(NMWellFile,'Deep',7000.,NMDTempWFile)
NMDInj.addMonthly(NMMonthlyFile,1000000,NMDTempMFile)
NMDInj.addDaily(NMDailyFile,100000,NMDTempDFile)

# Merge and process results

DeepWells=inj3.inj(NMDInj,TXDInj,'01-01-1980',DeepWellFile,DeepInjFile)
DeepWells.processRates(200000.,1,'12-16-2022',True)
DeepWells.output(DeepPrefix)

# Shallow
TXSInj=inj3.injTX(TXWellFile,'Shallow',7000.,TXSTempWFile)
TXSInj.addMonthly(TXMonthlyFile,1000000,TXSTempMFile)
TXSInj.addDaily(TXDailyFile,100000,TXSTempDFile)

NMSInj=inj3.injNM(NMWellFile,'Shallow',7000.,NMSTempWFile)
NMSInj.addMonthly(NMMonthlyFile,1000000,NMSTempMFile)
NMSInj.addDaily(NMDailyFile,100000,NMSTempDFile)


ShallowWells=inj3.inj(NMSInj,TXSInj,'01-01-1980',ShallowWellFile,ShallowInjFile)
ShallowWells.processRates(200000.,1,'12-16-2022',True)
ShallowWells.output(ShallowPrefix)

# New Mexico should look the same