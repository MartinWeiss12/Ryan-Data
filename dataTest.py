#!/usr/bin/env python3
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error 
from sklearn.svm import SVR
from scipy.stats import norm

data = pd.read_csv(r'/Users/martinweiss/Desktop/Python/usDensity.csv')

density = data['Density'].tolist()

densityArray = np.array(density)

states = data['States'].tolist()

statesArray = np.array(states)


length = len(data)



#define mean and standard deviation 
mean1 = np.mean(densityArray)
sd1 = 1

#define lower and upper bounds for x-axis
lower_bound = -4
upper_bound = 4

#create range of x-values from lower to upper bound in increments of .001
x = np.arange(lower_bound,upper_bound, 0.001)

#create range of y-values that correspond to normal pdf with mean1=0 and sd=1 
y = norm.pdf(x,0,1)

# build the plot
fig, ax = plt.subplots(figsize=(9,6))
ax.plot(x,y)

#define title for the plot
ax.set_title('Normal Gaussian Curve')

for i in range (length):
	
	densityPerState = density[i]
	
	state = statesArray[i]
	print(state)
	plt.annotate(state, (-2-(densityPerState/50), densityPerState/100))

#choose plot style and display the bell curve 
plt.style.use('fivethirtyeight')
plt.show()