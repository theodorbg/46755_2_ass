# Standard library imports
import os

# Third-party imports
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Local imports
from import_data import df_wind, df_price, df_conditions
from scenario_generator_loop import df_balancing_prices, df_in_sample_scenarios df_out_of_sample_scenarios

CAPACITY_WIND_FARM = 500 #MW
OFFER_PRICE_WIND_FARM = 0

# objective: formulate and solve optimization problem to determine its optimal offering strategy in terms of production quantity in the day-ahead market
# ANALYSIS SPAN: 24 HOURS
# MARKETS: day-ahead + balancing markets
# no reserve / intra-day markets

#uncertainty sources (hourly basis)
# 1.  1. Wind power production,
# 2. Day-ahead market price,
# 3. The real-time power system condition (whether the system experiences a supply deficit or excess)

# uncertainties assumed uncorrelated

# DATAFRAMES:
# df_wind: wind power production forecast (MW)
# df_price: day-ahead market price forecast (EUR/MWh)
# df_conditions: system conditions (binary, 1 = deficit, 0 = excess)
# df_balancing_prices: balancing market prices (EUR/MWh)

#df_in_sample_scenarios: in-sample scenarios (200 scenarios)
# df_out_of_sample_scenarios: out-of-sample scenarios (1400 scenarios)
