import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Uploading the dataset, creating dataframe (using pandas)
column_names= ['Shift','SNR','Data','Test acc','Test loss','Precision','Recall','F1']
df = pd.read_csv('Test1_data_file_N-10',sep=',', names=column_names, engine='python',index_col=False)

#-----------------------------------------------------------------------------#
# Comparison of time domain, magnitude spectrogram and rectangular spectrogram for various SNR
# for specific modulation type  
shift_key = 'phase'
df_X = df.loc[(df['Shift'] == shift_key)]
X=np.unique(np.array(df_X['SNR']))

df_Y1=df_X.loc[(df_X['Data'] == 0)]
Y1=df_Y1['Test acc']
df_Y2=df_X.loc[(df_X['Data'] == 1)]
Y2=df_Y2['Test acc']
df_Y3=df_X.loc[(df_X['Data'] == 2)]
Y3=df_Y3['Test acc']

sns.lineplot(x=df_Y1['SNR'],y=df_Y1['Test acc'],linewidth=4,color='r',label='Time-series',ci=100)
sns.lineplot(x=df_Y2['SNR'],y=df_Y2['Test acc'],linewidth=4,color='b',label='Magnitude spectrogram',ci=100)
sns.lineplot(x=df_Y3['SNR'],y=df_Y3['Test acc'],linewidth=4,color='g',label='Rectangular spectrogram',ci=100)

plt.xlabel('SNR [dB] ',fontsize=28)
plt.ylabel('Accuracy',fontsize=28)
plt.xticks(X,fontsize=20)
plt.yticks(np.linspace(0,1,21),fontsize=20)
plt.ylim(0.4,1.05)
plt.xlim(-21,21)
plt.legend(loc='lower right',fontsize=20)
plt.grid(linestyle='--',linewidth=2)
plt.tight_layout

#-----------------------------------------------------------------------------#