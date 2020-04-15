#Statistics project from Victor Carralero & Manuel Eljishi
#UPV Computer Engineering Degree 2020

'''
F1 = months,
F2 = days,
X1 = relative humidity,
X2 = wind,
X3 = temperature,
X4 = rain
'''
import sklearn
import pandas as pd
import numpy as np
from sklearn import preprocessing
import random
import matplotlib.pyplot as plt
import math
from scipy.stats import kurtosis, skew

df = pd.read_csv("forestfires.csv")
#print(df)
#print(df.skew(axis = 0, skipna = False))
le = preprocessing.LabelEncoder()

data = {
    "F1": {
        "Values": le.fit_transform(list(df["month"])),
        "Variance": 0.0,
        "Standard Deviation": 0.0,
        "Skewness": 0.0,
        "Kurtosis": 0.0
    },
    "F2": {
        "Values": le.fit_transform(list(df["day"])),
        "Variance": 0.0,
        "Standard Deviation": 0.0,
        "Skewness": 0.0,
        "Kurtosis": 0.0
    },
    "X1": {
        "Values": list(df["RH"]),
        "Variance": 0.0,
        "Standard Deviation": 0.0,
        "Skewness": 0.0,
        "Kurtosis": 0.0
    },
    "X2": {
        "Values": list(df["wind"]),
        "Variance": 0.0,
        "Standard Deviation": 0.0,
        "Skewness": 0.0,
        "Kurtosis": 0.0
    },
    "X3": {
        "Values": list(df["temp"]),
        "Variance": 0.0,
        "Standard Deviation": 0.0,
        "Skewness": 0.0,
        "Kurtosis": 0.0
    },
    "X4": {
        "Values": list(df["rain"]),
        "Variance": 0.0,
        "Standard Deviation": 0.0,
        "Skewness": 0.0,
        "Kurtosis": 0.0
    },
}

#Assign the values we loaded from file to the data dictionary
for key in data:
    #Access the value only once to store computing time
    values = data[key]["Values"]
    data[key]["Variance"] = np.var(values)
    data[key]["Standard Deviation"] = np.std(values)
    data[key]["Skewness"] = skew(values)
    data[key]["Kurtosis"] = kurtosis(values)
#print(data)

###Part 1###
#Building a frequency table with variable F2
# daysweek = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
# plt.table(cellText=)
# plt.gca().set(title='Frequency Histogram', ylabel='Frequency', xlabel = daysweek)
# plt.show()
freq_table = pd.crosstab(index=le.inverse_transform(data["F2"]["Values"]), columns="Count")
print(freq_table)