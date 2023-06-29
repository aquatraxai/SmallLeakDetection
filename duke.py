# Import libraries
import numpy as np
import pandas as pd
import random
import seaborn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import scipy
seaborn.set_palette(seaborn.color_palette())
import warnings; warnings.simplefilter('ignore')

# Read dataset
FILE_NAME = 'aquatrax_usage_data_new.csv'
df = pd.read_csv(FILE_NAME)

print(df.columns)
#print(df.head())
print(df['MeterId'].unique())

# Extract data from specific MeterID
METERID = 54624
df_account = df.loc[df['MeterId']==METERID]

print("len : {}".format(len(df_account)))
#print(df_account['Value'])
# Have they cleaned out rows where there are consecutive days
# Discrete Fourier Transform
df_account['FFT_W'] = [i/len(df_account) for i in range(len(df_account))]
df_account['FFT_Period'] = 1/df_account['FFT_W']
df_account['FFT'] = scipy.fft.fft(df_account['Value'])
df_account['FFTA'] = np.abs(df_account['FFT'])

# checking the consecutive days:
#df_account['Days'] = pd.to_datetime(df_account['Usagedate']).dt.date


# Plot frequency domain result
plt.title(f'Discrete Fourier Transform on Water Usage of Meter {METERID}')
g = seaborn.lineplot(data=df_account,x="FFT_Period",y="FFTA",color='b',size=0.01)
g.set(xlim=(0, 30))
g.set_xlabel(f'Signal Period (h)')
g.set_ylabel(f'Signal Amplitude')
legend = g.legend()
g.grid()
legend.remove()

fig = g.get_figure()
fig.savefig("/output/fourier.png") 

