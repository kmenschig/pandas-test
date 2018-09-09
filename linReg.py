import numpy as np
import pandas as pd
from pandas import DataFrame
import statsmodels.api as sm
import matplotlib.pyplot as plt

df = pd.read_csv('linReg.csv', header=None, sep=';', 
                                            names=['y','x'])
#print(df)

x = df['x']
#print(x.min())

x = sm.add_constant(x)

y = df['y']

model = sm.OLS(y, x)
fitted = model.fit()
x_pred = np.linspace(df['x'].min(), df['x'].max(), 10)
x_pred2 = sm.add_constant(x_pred)
y_pred = fitted.predict(x_pred)

print(x_pred, y_pred)

#fig, ax = plt.subplots(figsize=(8, 4))

#ax.scatter(x, y, alpha=0.5, color='orchid')
#fig.suptitle('Example Scatter Plot')
#fig.tight_layout(pad=2); 
#ax.grid(True)
#fig.savefig('filename1.png', dpi=125)




#ax.plot(x_pred, y_pred, '-', color='darkorchid', linewidth=2)
#fig.savefig('filename2.png', dpi=125)


