# read data from walleye_population

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# read data
walleye_population_abundance = pd.read_excel("walleye_population_data.xlsx")
walleye_population_abundance.set_index("Year", inplace=True)

fishing_mortality = pd.read_excel("walleye_population_fishing_mortality.xlsx")
fishing_mortality.set_index("Year", inplace=True)

catch_data = pd.read_excel("catch_data.xlsx")
catch_data.set_index("Year", inplace=True)

print(fishing_mortality)
print(walleye_population_abundance)
print(catch_data)

# estimate fishing mortality rate for each age class

mean_mortality = fishing_mortality.mean(axis=0)
print(mean_mortality)

# estimate natural mortality rate for each age class

non_fished_catch = walleye_population_abundance - catch_data
class_survival_rate = non_fished_catch / non_fished_catch.shift(1, axis=1)
class_survival_rate.replace(np.inf, np.nan, inplace=True)
print(class_survival_rate)

mean_natural_mortality = class_survival_rate.mean(axis=0)
print(mean_natural_mortality)



