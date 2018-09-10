#http://markthegraph.blogspot.com/2015/05/using-python-statsmodels-for-ols-#linear.html

from sys import platform

# Loading different matplotlib backend for macOS devices
# This needs to precede the loading of any other rendering packages.
if platform == 'darwin':
    import matplotlib as mpl
    mpl.use("TkAgg")

import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

INPUT_FILE = None

# TODO: Ideally, this should be handled by script arguments.
# Look into argument parsing for python
IS_LINEAR = input("Is this a linear function? Y/N: ").lower() == 'y'

# Because the only thing that changes is the file name
if IS_LINEAR:
    INPUT_FILE = 'expReg.csv'
else:
    INPUT_FILE = 'linReg.csv'

df = pd.read_csv(INPUT_FILE, header=None, sep=';',
                 names=['x','y'])

# Setting arrays equal to dataframe columns
y = df['y'].values
x = df['x'].values

if not IS_LINEAR:
    y = np.log(df['y'].values)
else:
    y = df['y'].values


fig, ax = plt.subplots(figsize=(8, 4))

if not IS_LINEAR:
    yplt = np.exp(y)
else:
    yplt = y

ax.scatter(x, yplt, alpha=0.5, color='orchid')

fig.suptitle('Funktion mit 95% Konfidenzinterval')

fig.tight_layout(pad=2);

ax.grid(True)

# TODO: pick a better name
fig.savefig('filename1.png', dpi=125)


x = sm.add_constant(x) # constant intercept term

# Model: y ~ x + c

model = sm.OLS(y, x)

fitted = model.fit()

x_pred = np.linspace(x.min(), x.max(), 50)

x_pred2 = sm.add_constant(x_pred)

y_pred = fitted.predict(x_pred2)

# if var=="n" or var=="N":
if not IS_LINEAR:
    yplt = np.exp(y_pred)
else:
    yplt = y_pred

ax.plot(x_pred, yplt, '-', color='darkorchid', linewidth=2)

# TODO: pick a better name
fig.savefig('filename2.png', dpi=125)

print(fitted.params)     # the estimated parameters for the regression line
print(fitted.summary())  # summary statistics for the regression

y_hat = fitted.predict(x) # x is

y_err = y - y_hat

mean_x = x.T[1].mean()

n = len(x)

dof = n - fitted.df_model - 1


t = stats.t.ppf(1-0.025, df=dof)

s_err = np.sum(np.power(y_err, 2))

conf = t * np.sqrt((s_err/(n-2))*(1.0/n + (np.power((x_pred-mean_x),2) /
         ((np.sum(np.power(x_pred,2))) - n*(np.power(mean_x,2))))))

upper = y_pred + abs(conf)
lower = y_pred - abs(conf)

if not IS_LINEAR:
    uplt = np.exp(upper)
    lplt = np.exp(lower)
else:
    uplt = upper
    lplt = lower

ax.fill_between(x_pred, lplt, uplt, color='#888888', alpha=0.4)

# TODO: rename this
fig.savefig('filename3.png', dpi=125)
