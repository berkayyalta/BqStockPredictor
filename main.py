from Classes import ModelSelection as ms
from Classes import Calculation as clc
from Classes import GenerateDatasets as gds

#This class can only be used to test the program for now. A user interface will be added in further versions.

#To find the variables to be used
"""
stocks = ['ASELS.IS','ARCLK.IS','AKBNK.IS','MGROS.IS','TTKOM.IS']
for stock in stocks:
    ms.SaveModel(ms, stock, 401)
"""

#Make predictions for all the models in Models.csv
results = clc.Predict(clc, 1201)
for result in results : 
    print(result)
    