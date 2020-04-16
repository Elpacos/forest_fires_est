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
import seaborn as sns
from sklearn import preprocessing
import random
import matplotlib.pyplot as plt
import math
from scipy.stats import kurtosis, skew, itemfreq, relfreq, iqr, probplot

df = pd.read_csv("forestfires.csv")
#print(df)
#print(df.skew(axis = 0, skipna = False))
le = preprocessing.LabelEncoder()
le2 = preprocessing.LabelEncoder()

data = {
    "F1": {
        "Values": le2.fit_transform(list(df["month"])),
        "Variance": 0.0,
        "Median": 0.0,
        "Average": 0.0,
        "Min": 0.0,
        "Max": 0.0,
        "Range": 0.0,
        "IQR":0.0,
        "Standard Deviation": 0.0,
        "Skewness": 0.0,
        "Kurtosis": 0.0
    },
    "F2": {
        "Values": le.fit_transform(list(df["day"])),
        "Variance": 0.0,
        "Median": 0.0,
        "Average": 0.0,
        "Min": 0.0,
        "Max": 0.0,
        "Range": 0.0,
        "IQR":0.0,
        "Standard Deviation": 0.0,
        "Skewness": 0.0,
        "Kurtosis": 0.0
    },
    "X1": {
        "Values": list(df["RH"]),
        "Variance": 0.0,
        "Median": 0.0,
        "Average": 0.0,
        "Min": 0.0,
        "Max": 0.0,
        "Range": 0.0,
        "IQR":0.0,
        "Standard Deviation": 0.0,
        "Skewness": 0.0,
        "Kurtosis": 0.0
    },
    "X2": {
        "Values": list(df["wind"]),
        "Variance": 0.0,
        "Median": 0.0,
        "Average": 0.0,
        "Min": 0.0,
        "Max": 0.0,
        "Range": 0.0,
        "IQR":0.0,
        "Standard Deviation": 0.0,
        "Skewness": 0.0,
        "Kurtosis": 0.0
    },
    "X3": {
        "Values": list(df["temp"]),
        "Variance": 0.0,
        "Median": 0.0,
        "Average": 0.0,
        "Min": 0.0,
        "Max": 0.0,
        "Range": 0.0,
        "IQR":0.0,
        "Standard Deviation": 0.0,
        "Skewness": 0.0,
        "Kurtosis": 0.0
    },
    "X4": {
        "Values": list(df["rain"]),
        "Variance": 0.0,
        "Median": 0.0,
        "Average": 0.0,
        "Min": 0.0,
        "Max": 0.0,
        "Range": 0.0,
        "IQR": 0.0,
        "Standard Deviation": 0.0,
        "Skewness": 0.0,
        "Kurtosis": 0.0
    },
}
#Assign the values we loaded from file to the data dictionary
for key in data:
    #Access the value only once to store computing time
    if key is "X1" or key is "X2" or key is "X3" or key is "X4":
        data[key]["Values"] = np.sort(data[key]["Values"])
    data[key]["Average"] = np.average(data[key]["Values"])
    data[key]["Min"] = np.amin(data[key]["Values"])
    data[key]["Max"] = np.amax(data[key]["Values"])
    data[key]["Range"] = data[key]["Max"] - data[key]["Min"]
    data[key]["Median"] = np.median(data[key]["Values"])
    data[key]["IQR"] = iqr(data[key]["Values"])
    data[key]["Variance"] = np.var(data[key]["Values"])
    data[key]["Standard Deviation"] = np.std(data[key]["Values"])
    data[key]["Skewness"] = skew(data[key]["Values"])
    data[key]["Kurtosis"] = kurtosis(data[key]["Values"])
    #print(data[key])

#Sort the arrays of days and months
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
'''y_posf2 = np.arange(len(dayArray[:,0]))
plt.bar(y_posf2,dayArray[:,1].astype(np.int))
plt.xticks(y_posf2, dayArray[:,0])
plt.ylabel('Number of fires started')
plt.xlabel('Days of the week')
plt.title('Fires stared per weekday')
plt.show()'''



##Part 3###
#Build a piechart with variable F1
'''legendf1_1 = np.core.defchararray.add(monthArray[:,0], [":   "])
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
plt.show()'''


###Part 4###
#Build a contingency table(cross frequency) with variables F1 & F2
'''tab = pd.crosstab(df['month'], df['day'])
print(tab)'''

###Part 5###
#Write on a table the max, min, range, interquartile range, mean, median
#variance, std, asymmetry coefficient and kurtosis coefficent
#lines 110 - 121

###Part 6###
#Build a histogram and a box-whisker plot of variable X1
"""plt.hist(data["X1"]["Values"], bins=21, align='mid' )
plt.xticks([0,10,20,30,40,50,60,70,80,90,100])
plt.ylabel("Number of occurrences",fontsize=18, rotation=90)
plt.xlabel("Relative Humidity")
plt.show()"""

"""fig6, ax6 = plt.subplots()
ax6.set_title('X1 variable')
ax6.boxplot(data['X1']['Values'], vert=False)
plt.show()"""

###Part 7###
#Build a histogram and a normal probabilistic paper
""" plt.hist(data["X2"]["Values"], bins=21, align='mid' )
plt.ylabel("Number of occurrences",fontsize=18, rotation=90)
plt.xlabel("Wind speed in km/h:")
plt.show() """
"""sns.set(rc={'figure.figsize':(10, 9)})
sns.set_context('talk')
probplot(data["X2"]["Values"], dist="norm", fit=True, rvalue=True, plot=plt)
plt.xlabel("Theoretical quantiles\nMy interpretation: standard deviations", labelpad=12)
plt.ylabel("Wind speed in km/h")
plt.title("Probability Plot to Compare normal_distr_values to Perfectly Normal Distribution", y=1.015)
plt.show()"""

###Part 8###
#Build a histogram with variable X3, and a multiple box-whisker plot
"""plt.hist(data["X3"]["Values"], bins=21, align='mid' )
plt.ylabel("Number of occurrences",fontsize=18, rotation=90)
plt.xlabel("Temperature")
plt.show()"""

month_temp = np.array([["jan",[] ],["feb",[] ],["mar",[] ],["apr", []],["may", []],["jun", []],["jul", []],["aug", []],["sep", []],["oct", []],["nov",[] ],["dec",[] ]])
for j, i in df.iterrows():
    if i["month"] == "jan":
        month_temp[0][1].append(i["temp"])
    elif i["month"] == "feb":
        month_temp[1][1].append(i["temp"])
    elif i["month"] == "mar":
        month_temp[2][1].append(i["temp"])
    elif i["month"] == "apr":
        month_temp[3][1].append(i["temp"])
    elif i["month"] == "may":
        month_temp[4][1].append(i["temp"])
    elif i["month"] == "jun":
        month_temp[5][1].append(i["temp"])
    elif i["month"] == "jul":
        month_temp[6][1].append(i["temp"])
    elif i["month"] == "aug":
        month_temp[7][1].append(i["temp"])
    elif i["month"] == "sep":
        month_temp[8][1].append(i["temp"])
    elif i["month"] == "oct":
        month_temp[9][1].append(i["temp"])
    elif i["month"] == "nov":
        month_temp[10][1].append(i["temp"])
    else:
        month_temp[11][1].append(i["temp"])

plt.figure()
monthsf2 = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep","oct","nov","dec"]
jan = plt.boxplot(month_temp[0][1], positions= [0])
feb = plt.boxplot(month_temp[1][1], positions= [1] )
mar = plt.boxplot(month_temp[2][1], positions= [2])
apr = plt.boxplot(month_temp[3][1], positions= [3])
may = plt.boxplot(month_temp[4][1], positions= [4])
jun = plt.boxplot(month_temp[5][1], positions= [5])
jul = plt.boxplot(month_temp[6][1], positions= [6])
aug = plt.boxplot(month_temp[7][1], positions= [7])
sep = plt.boxplot(month_temp[8][1], positions= [8])
octo = plt.boxplot(month_temp[9][1], positions= [9])
nov = plt.boxplot(month_temp[10][1], positions= [10])
dec = plt.boxplot(month_temp[11][1], positions= [11])
plt.xticks(range(0, len(monthsf2)), monthsf2)
plt.show()