import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib notebook

np.random.seed(12345)

df = pd.DataFrame([np.random.normal(33500,150000,3650), 
                   np.random.normal(41000,90000,3650), 
                   np.random.normal(41000,120000,3650), 
                   np.random.normal(48000,55000,3650)], 
                  index=[1992,1993,1994,1995])
# Transpose:
df=df.transpose()
# Long Format:
df=pd.melt(df)

# Aggregate Mean, Std and number Observations per Year:
data=pd.DataFrame({'mean':df.groupby('variable').mean()['value'],
    'std':df.groupby('variable').std()['value'],
    'n':df.groupby('variable').count()['value']},
    index=[1992,1993,1994,1995])

# Calculate the Standard Error of the Mean:
data['SE']=data['std']/np.sqrt(data['n'])

# Calculate 95 %-Confidence Interval Limits:
data['LowerCI']=data['mean']-1.96*data['SE']
data['UpperCI']=data['mean']+1.96*data['SE']

# In[ ]:

#### Gradient solution:
# Goal: Colormap in Relation to the treshhold value. 
# How high is the probability to be higher than the treshhold value?
    
# Position of X-Labels (Years):
pos = np.arange(len(data.index.values))
# Starting Reference of Votes:
Reference=38000

# Range of the Confidence Interval for a Given Year:
data['RangeCI']=data['UpperCI']-data['LowerCI']
# Absolute Deviation of Upper CI Limit from Reference Value:
data['Diff']=data['UpperCI']-Reference

# Colormap according to Relation to Reference Value:
colors = []
for Lower,Upper in zip(data['LowerCI'],data['UpperCI']):    
    if Lower>Reference:
        # -> LowerCI is above Reference, above Reference for a High Probability
        colors.append(1)
    else:
        if Lower<Reference and Upper>Reference:
            # Reference Value Inside Confidence Interval
            # Color Depends on: (Upper-Reference)/(Upper-Lower)
            # The Nearer Reference to the Upper Limit Compared to 
            # Overall Range of Interval, the lower the Value gets
            
            # Deviation Reference from Upper Limit:
            Dev=Upper-Reference         
            colors.append(Dev/data.loc[data['UpperCI'] == Upper, 'RangeCI'].values[0])
        else:
            colors.append(0)
        
# Import Colormap:
import matplotlib.cm as cm
import matplotlib.colorbar as colorbar
import matplotlib.colors as col
# Blue to Red Colors:
cmap = cm.ScalarMappable(norm=col.Normalize(0,1),cmap='RdBu_r')

# Figure Mapping
fig=plt.figure()
plt.bar(pos,data['mean'],width=1,
        color=cmap.to_rgba(colors,alpha=0.75),linewidth=2,yerr=data['SE']*1.96,
        error_kw=dict(ecolor='black', lw=2, capsize=20, capthick=2),picker=10)
plt.xticks(pos, data.index.values, alpha=0.8)
plt.ylim(0,55000)
plt.tick_params(top='off', bottom='off', labelbottom='on')
plt.axhline(y=Reference,color='grey',lw=3)
plt.title('You Have Choosen as a Reference Line {:5.0f} Votes \n Click Inside the Graph to Change the Reference'.format(round(Reference,0)))
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
# BarLegend
ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.05])
cb = colorbar.ColorbarBase(ax1,cmap='RdBu_r',
                                norm=col.Normalize(0,1),
                                orientation='horizontal',
                                ticks=[0.1,0.9])
cb.set_ticklabels(['Low Probability \n Above Reference','High Probability \n Above Reference'])
plt.show()

# Define oneclick to update plot via new choosen reference
def onclick(event):
    plt.clf()
    Reference=event.ydata
    
    # Update Colors:
    colors = []
    for Lower,Upper in zip(data['LowerCI'],data['UpperCI']):    
        if Lower>Reference:
            colors.append(1)
        else:
            if Lower<Reference and Upper>Reference:
                Dev=Upper-Reference         
                colors.append(Dev/data.loc[data['UpperCI'] == Upper, 'RangeCI'].values[0])
            else:
                colors.append(0)
                
    cmap = cm.ScalarMappable(norm=col.Normalize(0,1),cmap='RdBu_r')
    plt.bar(pos,data['mean'],width=1,
            color=cmap.to_rgba(colors,alpha=0.75),linewidth=2,yerr=data['SE']*1.96,
            error_kw=dict(ecolor='black', lw=2, capsize=20, capthick=2),picker=10)
    plt.xticks(pos, data.index.values, alpha=0.8)
    plt.axhline(y=Reference,color='grey',lw=3)
    plt.title('You Have Choosen as a Reference Line {:5.0f} Votes \n Click Inside the Graph to Change the Reference'.format(round(event.ydata,0)))
    plt.ylim(0,55000)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
    # BarLegend
    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.05])
    cb = colorbar.ColorbarBase(ax1,cmap='RdBu_r',
                                    norm=col.Normalize(0,1),
                                    orientation='horizontal',
                                    ticks=[0.1,0.9])
    cb.set_ticklabels(['Low Probability \n Above Reference','High Probability \n Above Reference'])
    fig.canvas.draw()
    
fig.canvas.mpl_connect('button_press_event', onclick)


# Display a textbox showing the selected y value except the default case when y = 0.
#    if y_level != 0:
#        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=1, color = 'black')
#        plt.text(-1, y_level, str( round(y_level,2) ) , ha="center", va="center", size=9, bbox=bbox_props) 
    