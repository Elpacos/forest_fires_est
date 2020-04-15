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
from scipy.stats import kurtosis, skew, itemfreq, relfreq

df = pd.read_csv("forestfires.csv")
#print(df)
#print(df.skew(axis = 0, skipna = False))
le = preprocessing.LabelEncoder()
le2 = preprocessing.LabelEncoder()

data = {
    "F1": {
        "Values": le2.fit_transform(list(df["month"])),
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

#Sort the arrays of days and months so
freq_tablef2 = itemfreq(le.inverse_transform(data["F2"]["Values"]))
daysweekf2 = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
sorted_daysf2 = sorted(freq_tablef2, key=lambda x: daysweekf2.index(x[0]))
dayArray = np.array(sorted_daysf2)

freq_tablef1 = itemfreq(le2.inverse_transform(data["F1"]["Values"]))
monthsf1 = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep","oct","nov","dec"]
sorted_monthsf1 = sorted(freq_tablef1, key=lambda x: monthsf1.index(x[0]))
monthArray = np.array(sorted_monthsf1)

###Part 1###
#Building a frequency table with variable F2
#By this method we can compute the frequency of the days the fires were started
"""
print(freq_tablef2)"""
#On this method we create an empty list that will store the relative frequency values for each day
#that is the number of fires started on that day divided by the total number of fires registered
""" rel_freqf2 = []
for i in freq_tablef2:
    rel_freqf2.append([i[0], i[1].astype(np.int)/ len(data["F2"]["Values"])])
print(rel_freqf2) """

###Part 2###
#Building a barchart with the variable F2

y_posf2 = np.arange(len(dayArray[:,0]))
plt.bar(y_posf2,dayArray[:,1].astype(np.int))
plt.xticks(y_posf2, dayArray[:,0])
plt.ylabel('Number of fires started')
plt.xlabel('Days of the week')
plt.title('Fires stared per weekday')
plt.show()



##Part 3###
#Build a piechart with variable F1

legendf1_1 = np.core.defchararray.add(monthArray[:,0], [":   "])
percentagef1 = monthArray[:, 1].astype(np.int)/len(data["F1"]["Values"])
percentagef1 = percentagef1 * 100
percentagef1 = np.around(percentagef1, decimals=2)
percentagef1 = percentagef1.astype(np.str)
print(percentagef1)
legendf1 = np.core.defchararray.add(legendf1_1,percentagef1)
legendf1 = np.core.defchararray.add(legendf1, "%")
print(legendf1)
patches, texts = plt.pie(monthArray[:,1], startangle=90)
plt.legend(patches, legendf1, loc="best")
plt.axis('equal')
plt.tight_layout()
plt.show()


###Part 4###
#Build a contingency table(cross frequency) with variables F1 & F2
tab = pd.crosstab(df['month'], df['day'])
print(tab)