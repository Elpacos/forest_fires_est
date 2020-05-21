#Statistics project from Victor Martínez & Manuel Eljishi
#UPV Computer Engineering Degree 2020

'''
F1 = months,
F2 = days,
X1 = relative humidity,
X2 = wind,
X3 = temperature,
X4 = rain,
X5 = x-axis spatial coordinate within the Montesinho park map
X6 = y-axis spatial coordinate within the Montesinho park map
'''
import scipy.stats as st
import random
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
    "X5": {
        "Values": list(df["X"]),
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
    "X6": {
        "Values": list(df["Y"]),
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
        data[key]["Values"] = data[key]["Values"]
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
'''freq_tablef2 = itemfreq(le.inverse_transform(data["F2"]["Values"]))
daysweekf2 = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
sorted_daysf2 = sorted(freq_tablef2, key=lambda x: daysweekf2.index(x[0]))
dayArray = np.array(sorted_daysf2)

freq_tablef1 = itemfreq(le2.inverse_transform(data["F1"]["Values"]))
monthsf1 = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep","oct","nov","dec"]
sorted_monthsf1 = sorted(freq_tablef1, key=lambda x: monthsf1.index(x[0]))
monthArray = np.array(sorted_monthsf1)'''

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
plt.show()
sns.set(rc={'figure.figsize':(10, 9)})
sns.set_context('talk')
probplot(data["X2"]["Values"], dist="norm", fit=True, rvalue=True, plot=plt)
plt.xlabel("Theoretical quantiles\nInterpretation: standard deviations", labelpad=12)
plt.ylabel("Wind speed in km/h")
plt.title("X2 variable", y=1.015)
plt.show()"""

###Part 8###
#Build a histogram with variable X3, and a multiple box-whisker plot
"""plt.hist(data["X3"]["Values"], bins=21, align='mid' )
plt.ylabel("Number of occurrences",fontsize=18, rotation=90)
plt.xlabel("Temperature")
plt.show()"""

"""month_temp = np.array([["jan",[] ],["feb",[] ],["mar",[] ],["apr", []],["may", []],["jun", []],["jul", []],["aug", []],["sep", []],["oct", []],["nov",[] ],["dec",[] ]])
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
plt.show()"""


###Part 9###
#Build a histogram, boxplot and probabilistic paper and choose the best one
"""
plt.hist(data["X4"]["Values"], bins=21, align='mid' )
plt.ylabel("Number of occurrences",fontsize=18, rotation=90)
plt.xlabel("Rain in mm/m2")
plt.show()

fig6, ax6 = plt.subplots()
ax6.set_title('X4 variable')
ax6.boxplot(data['X4']['Values'], vert=False)
plt.show()

sns.set(rc={'figure.figsize':(10, 9)})
sns.set_context('talk')
probplot(data["X4"]["Values"], dist="norm", fit=True, rvalue=True, plot=plt)
plt.xlabel("Theoretical quantiles\nInterpretation: standard deviations", labelpad=12)
plt.ylabel("Rain in mm/m2")
plt.title("X4 variable", y=1.015)
plt.show()"""

###Part 10###
#Multiply half of the values of X1 by 2 and plot a histogram
"""shuffled = data["X1"]["Values"]
np.random.shuffle(shuffled)
for i in range(int(len(shuffled)/2)):
    shuffled[i] = shuffled[i] * 2
plt.hist(shuffled, bins=21, align='mid' )
plt.ylabel("Number of occurrences",fontsize=18, rotation=90)
plt.xlabel("Relative humidity multiplied by 2", fontsize=18)
plt.show()"""

###Part 11###
#Work with the discrete variables (X5 & X6)
'''print("X5", data["X5"]["Skewness"],data["X5"]["Kurtosis"])
print("X6", data["X6"]["Skewness"],data["X6"]["Kurtosis"])
plt.hist(data["X5"]["Values"], bins=21, align='mid' )
plt.ylabel("Number of occurrences",fontsize=18, rotation=90)
plt.xlabel("X5", fontsize=18)
plt.show()

plt.hist(data["X6"]["Values"], bins=21, align='mid' )
plt.ylabel("Number of occurrences",fontsize=18, rotation=90)
plt.xlabel("X6", fontsize=18)
plt.show()'''

###Part 12###
#Work with the continous variables (X1, X2, X3, X4)

#print(data["X1"], "\n", data["X2"], "\n", data["X3"], "\n" ,data["X4"], "\n")

### Part 13 ###
# Log() the continue variables X1, X2, X3, X4

"""X1log = np.array(data["X1"]["Values"])
X1log = np.log(X1log)

plt.hist(X1log, bins=21, align='mid' )
plt.ylabel("Number of occurrences",fontsize=18, rotation=90)
plt.xlabel("X1log", fontsize=18)
plt.show()

X2log = np.array(data["X2"]["Values"])
X2log = np.log(X2log)
plt.hist(X2log, bins=21, align='mid' )
plt.ylabel("Number of occurrences",fontsize=18, rotation=90)
plt.xlabel("X2log", fontsize=18)
plt.show()

X3log = np.array(data["X3"]["Values"])
X3log = np.log(X3log)

plt.hist(X3log, bins=21, align='mid' )
plt.ylabel("Number of occurrences",fontsize=18, rotation=90)
plt.xlabel("X3log", fontsize=18)
plt.show()"""


### Part 15 ###
#We sum variables of two normal distributions per pairs X1 and X2
# normalsum = np.add(data["X1"]["Values"], data["X2"]["Values"])
# plt.hist(normalsum, bins=21, align='mid' )
# plt.ylabel("Number of occurrences",fontsize=18, rotation=90)
# plt.xlabel("X3log", fontsize=18)
# plt.show()

""" print(data["X1"]["Standard Deviation"] * data["X1"]["Standard Deviation"] + data["X2"]["Standard Deviation"] * data["X2"]["Standard Deviation"] ) """


### Part 16 ###
#Get a 5 random value sample of the X1 poblation

random_array = []

for i in range(5):
    aux = random.randint(0, (len(data["X1"]["Values"])))
    random_array.append(data["X1"]["Values"][aux])

a = np.array(random_array)
mean_random = np.mean(a)
interval = st.t.interval(0.95, len(a)-1, loc=mean_random, scale=st.sem(a))
print(interval)
print(mean_random)