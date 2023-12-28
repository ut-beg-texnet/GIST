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

TXDTempWFile='/data/xom/seismicity/bcurry/GIST/TestTXWellsDeep10.csv'
TXDTempMFile='/data/xom/seismicity/bcurry/GIST/TestTXMonthlyInjDeep10.csv'
TXDTempDFile='/data/xom/seismicity/bcurry/GIST/TestTXDailyInjDeep10.csv'
TXSTempWFile='/data/xom/seismicity/bcurry/GIST/TestTXWellsShallow10.csv'
TXSTempMFile='/data/xom/seismicity/bcurry/GIST/TestTXMonthlyInjShallow10.csv'
TXSTempDFile='/data/xom/seismicity/bcurry/GIST/TestTXDailyInjShallow10.csv'

NMWellFile='/data/xom/seismicity/shared/B3/NM/InjectionWell.csv'
NMMonthlyFile='/data/xom/seismicity/shared/B3/NM/MonthlyInjection.csv'
NMDailyFile='/data/xom/seismicity/shared/B3/NM/DailyInjection.csv'

NMDTempWFile='/data/xom/seismicity/bcurry/GIST/TestNMWellsDeep10.csv'
NMDTempMFile='/data/xom/seismicity/bcurry/GIST/TestNMMonthlyInjDeep10.csv'
NMDTempDFile='/data/xom/seismicity/bcurry/GIST/TestNMDailyInjDeep10.csv'
NMSTempWFile='/data/xom/seismicity/bcurry/GIST/TestNMWellsShallow10.csv'
NMSTempMFile='/data/xom/seismicity/bcurry/GIST/TestNMMonthlyInjShallow10.csv'
NMSTempDFile='/data/xom/seismicity/bcurry/GIST/TestNMDailyInjShallow10.csv'

DeepWellFile='/data/xom/seismicity/bcurry/GIST/testDeep10.csv'
DeepInjFile='/data/xom/seismicity/bcurry/GIST/testDeepInj10.csv'
DeepPrefix='/data/xom/seismicity/bcurry/GIST/testDeep10'

ShallowWellFile='/data/xom/seismicity/bcurry/GIST/testShallow10.csv'
ShallowInjFile='/data/xom/seismicity/bcurry/GIST/testShallowInj10.csv'
ShallowPrefix='/data/xom/seismicity/bcurry/GIST/testShallow10'
# First deal with Deep
# Texas
TXDInj=inj3.injTX(TXWellFile,'Deep',7000.,TXDTempWFile,1)
TXDInj.addMonthly(TXMonthlyFile,1000000,TXDTempMFile,1)
TXDInj.addDaily(TXDailyFile,100000,TXDTempDFile,1)

# New Mexico
NMDInj=inj3.injNM(NMWellFile,'Deep',7000.,NMDTempWFile,1)
NMDInj.addMonthly(NMMonthlyFile,1000000,NMDTempMFile,1)
NMDInj.addDaily(NMDailyFile,100000,NMDTempDFile,1)

# Merge and process results

DeepWells=inj3.inj(NMDInj,TXDInj,'01-01-1970',DeepWellFile,DeepInjFile,1)
DeepWells.processRates(200000.,10,'12-16-2022',True,1)
DeepWells.output(DeepPrefix,1)

# Shallow
TXSInj=inj3.injTX(TXWellFile,'Shallow',7000.,TXSTempWFile,1)
TXSInj.addMonthly(TXMonthlyFile,1000000,TXSTempMFile,1)
TXSInj.addDaily(TXDailyFile,100000,TXSTempDFile,1)

NMSInj=inj3.injNM(NMWellFile,'Shallow',7000.,NMSTempWFile,1)
NMSInj.addMonthly(NMMonthlyFile,1000000,NMSTempMFile,1)
NMSInj.addDaily(NMDailyFile,100000,NMSTempDFile,1)


ShallowWells=inj3.inj(NMSInj,TXSInj,'01-01-1970',ShallowWellFile,ShallowInjFile,1)
ShallowWells.processRates(200000.,10,'12-16-2022',True,1)
ShallowWells.output(ShallowPrefix,1)
