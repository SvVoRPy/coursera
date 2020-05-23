import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.dates import DateFormatter
WORKINGDIRECTORY="C:/Users/SvenV/Desktop/Coursera Kurse/Python Spezialization II/Assignments/Assignment 2"

df=pd.read_csv(WORKINGDIRECTORY+'/fb441e62df2d58994928907a91895ec62c2c42e6cd075c2700843b89.csv'.format(400))

# Convert Date:
df['DateOld'] = df['Date'] =pd.to_datetime(df['Date'])
# Extract Year:
df['Year'] =df['Date'].dt.year
# Extract Month/Day Combination:
df['Date'] =df['Date'].dt.strftime('%m-%d')
# Exclude 02-29:
df=df[df['Date']!="02-29"]

# Tenth of Degree in Degree Celcius:
df['Data_Value']=df['Data_Value']/10

df.index=df['DateOld']

# Calculate Max and Min by Day of Year for 2005 to 2014
df_TMAX=df[(df['Element']=="TMAX") & (df['Year']<2015)].copy().groupby('Date')['Data_Value'].max().reset_index()
df_TMIN=df[(df['Element']=="TMIN") & (df['Year']<2015)].copy().groupby('Date')['Data_Value'].min().reset_index()

# Has 2015 been a record?
max_2015=pd.merge(df[(df['Element']=="TMAX") & (df['Year']==2015)][['Date','Data_Value']],df_TMAX,how='inner',on='Date')
min_2015=pd.merge(df[(df['Element']=="TMIN") & (df['Year']==2015)][['Date','Data_Value']],df_TMIN,how='inner',on='Date')
# Dummy 1 if Record in 2015:
max_2015['Rec'] = np.where(max_2015['Data_Value_x']>max_2015['Data_Value_y'],1,0)
min_2015['Rec'] = np.where(min_2015['Data_Value_x']<min_2015['Data_Value_y'],1,0)

# Get synthetic date format for plotting reasons:
df_TMAX['Date']=pd.to_datetime("2017-"+df_TMAX['Date'],format="%Y-%m-%d")
df_TMIN['Date']=pd.to_datetime("2017-"+df_TMIN['Date'],format="%Y-%m-%d")

max_2015['Date']=pd.to_datetime("2017-"+max_2015['Date'],format="%Y-%m-%d")
min_2015['Date']=pd.to_datetime("2017-"+min_2015['Date'],format="%Y-%m-%d")

# Date as Index:
df_TMAX.index = pd.date_range(start='2017-1-1', end='2017-12-31', freq='d')
df_TMIN.index = pd.date_range(start='2017-1-1', end='2017-12-31', freq='d')

#### Plot generation:
plt.figure(figsize=(11, 7))

# Line Maximum
plt.plot(df_TMAX['Data_Value'],'r',alpha=0.4)
# Line Minimum:
plt.plot(df_TMIN['Data_Value'],'b',alpha=0.4)
# Scatter Maximum:
plt.plot(max_2015[max_2015['Rec']==1]['Date'],max_2015[max_2015['Rec']==1]['Data_Value_x'],'ro',markersize=2.5)
# Scatter Minimum:
plt.plot(min_2015[min_2015['Rec']==1]['Date'],min_2015[min_2015['Rec']==1]['Data_Value_x'],'bo',markersize=2.5)
# Shaded Region:
plt.fill_between(df_TMIN.index,df_TMIN['Data_Value'], df_TMAX['Data_Value'],facecolor='lightgrey',alpha=0.25,linewidth=0.25)

# Legend: 
plt.legend(loc=8,labels=['Max 2005-2014','Min 2005-2014','Overall Record in 2015','Overall Record in 2015'], frameon=False, title=False, ncol=2,numpoints=1)
# Format Date:
plt.gcf().axes[0].xaxis.set_major_formatter(DateFormatter('%b'))

# Title:
plt.title("Maximum and Minimum Temperature Records \n for Ann Arbor, Michigan, United States from 2005 to 2014 and vs. 2015")

# Get rid of Ticks:
plt.tick_params(top='off', bottom='off', labelleft='off', labelbottom='on')

# First y-Axis for Degrees Celcius
plt.ylim([-45,45])
plt.yticks([-40,-20,0,20,40])
ax = plt.gca()
ax.set_yticklabels(['-40 °C','-20 °C','0 °C','20 °C','40 °C'])

# Twin Axis for Fahrenheit:
ax2 = ax.twinx()
plt.ylim([-45,45])
plt.yticks([-40,-20,0,20,40])
ax2.set_yticklabels(['-40 °F','-4 °F','32 °F','68 °F','104 °F'])

# Rotation Labels x-Axis
for item in ax.get_xticklabels():
    item.set_rotation(90)
    
# Change Color Splines
for spine in ax.spines:
    ax.spines[spine].set_color("lightgrey")

# Save Plot:    
plt.savefig(WORKINGDIRECTORY,dpi=200)

