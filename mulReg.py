import numpy as np
import pandas as pd
from pandas import DataFrame
import statsmodels.api as sm
import matplotlib.pyplot as plt

df = pd.read_csv('summaryResultsShort.csv', header=None, sep=';', 
                                            names=['y','x1','x2'])
df['x1^2'] = (df['x1']) * (df['x1'])
df['x2^2'] = (df['x2']) * (df['x2'])
df['x1x2'] = (df['x1']) * (df['x2'])

X = df[['x1','x2','x1^2', 'x2^2', 'x1x2']]

#X = sm.add_constant(X)

y = df['y']

model = sm.OLS(y, X).fit()

predictions = model.predict(X) # make the predictions by the model
df['predictions'] = predictions

# Print out the statistics
print(model.summary())
print()
print("p-Values:")
print(model.pvalues)
#print(model.conf_int(alpha=0.05, cols=None)

#print(predictions)

#data = df[['y','predictions']]
#print(data)
#plt.plot(data)
#plt.show()
