# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 13:59:30 2017

@author: SvenV
"""


# coding: utf-8

# # Assignment 4
# 
# Before working on this assignment please read these instructions fully. In the submission area, you will notice that you can click the link to **Preview the Grading** for each step of the assignment. This is the criteria that will be used for peer grading. Please familiarize yourself with the criteria before beginning the assignment.
# 
# This assignment requires that you to find **at least** two datasets on the web which are related, and that you visualize these datasets to answer a question with the broad topic of **religious events or traditions** (see below) for the region of **Ann Arbor, Michigan, United States**, or **United States** more broadly.
# 
# You can merge these datasets with data from different regions if you like! For instance, you might want to compare **Ann Arbor, Michigan, United States** to Ann Arbor, USA. In that case at least one source file must be about **Ann Arbor, Michigan, United States**.
# 
# You are welcome to choose datasets at your discretion, but keep in mind **they will be shared with your peers**, so choose appropriate datasets. Sensitive, confidential, illicit, and proprietary materials are not good choices for datasets for this assignment. You are welcome to upload datasets of your own as well, and link to them using a third party repository such as github, bitbucket, pastebin, etc. Please be aware of the Coursera terms of service with respect to intellectual property.
# 
# Also, you are welcome to preserve data in its original language, but for the purposes of grading you should provide english translations. You are welcome to provide multiple visuals in different languages if you would like!
# 
# As this assignment is for the whole course, you must incorporate principles discussed in the first week, such as having as high data-ink ratio (Tufte) and aligning with Cairoâ€™s principles of truth, beauty, function, and insight.
# 
# Here are the assignment instructions:
# 
#  * State the region and the domain category that your data sets are about (e.g., **Ann Arbor, Michigan, United States** and **religious events or traditions**).
#  * You must state a question about the domain category and region that you identified as being interesting.
#  * You must provide at least two links to available datasets. These could be links to files such as CSV or Excel files, or links to websites which might have data in tabular form, such as Wikipedia pages.
#  * You must upload an image which addresses the research question you stated. In addition to addressing the question, this visual should follow Cairo's principles of truthfulness, functionality, beauty, and insightfulness.
#  * You must contribute a short (1-2 paragraph) written justification of how your visualization addresses your stated research question.
# 
# What do we mean by **religious events or traditions**?  For this category you might consider calendar events, demographic data about religion in the region and neighboring regions, participation in religious events, or how religious events relate to political events, social movements, or historical events.
# 
# ## Tips
# * Wikipedia is an excellent source of data, and I strongly encourage you to explore it for new data sources.
# * Many governments run open data initiatives at the city, region, and country levels, and these are wonderful resources for localized data sources.
# * Several international agencies, such as the [United Nations](http://data.un.org/), the [World Bank](http://data.worldbank.org/), the [Global Open Data Index](http://index.okfn.org/place/) are other great places to look for data.
# * This assignment requires you to convert and clean datafiles. Check out the discussion forums for tips on how to do this from various sources, and share your successes with your fellow students!
# 
# ## Example
# Looking for an example? Here's what our course assistant put together for the **Ann Arbor, MI, USA** area using **sports and athletics** as the topic. [Example Solution File](./readonly/Assignment4_example.pdf)

#import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Polygon


    
att = pd.read_table("attendance.txt", header= None, usecols= [1,2], names = ["State","Attendance"])
att["State"]=att["State"].apply(lambda x: x.strip())

pop = pd.read_table("population.txt",usecols=[1,2],header=None, names=["State","Population"])
pop["State"] = pop["State"].apply(lambda x: x.strip())


NofPlaces = pd.read_csv("2010.csv", skiprows=4, usecols=[0,5])

df = NofPlaces.merge(pop,how="left",on="State")
df = df.merge(att,how="left",on="State")
df.Attendance = df.Attendance.apply(lambda x: int(x[:-1])/100)
df.Population= df.Population.apply(lambda x: 
    int(x.replace(",","")) if isinstance(x,str) else x)
df["PeoplePerVenue"] = df.Population*df.Attendance/df.Congregations
df.sort_values(by="PeoplePerVenue",ascending=False)

fig, [ax1, ax2] = plt.subplots(nrows=1,ncols=2,figsize=(20,20))
#norm = mpl.colors.Normalize(vmin=0, vmax=600)

cmap = plt.cm.get_cmap(name="OrRd")
norm = mpl.colors.BoundaryNorm(np.arange(0,650,100),cmap.N)
m = cm.ScalarMappable(norm=norm, cmap="OrRd")
m.set_array(np.arange(0,600,10))

divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(m,ax=ax1,cax=cax)

map = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
        projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
map.readshapefile('st99_d00', name='states', drawbounds=True, ax=ax1)
map.readshapefile('st99_d00', name='states', drawbounds=True, ax=ax2)

state_names = []
for shape_dict in map.states_info:
    state_names.append(shape_dict['NAME'])

for state in df.State:
    indices = [i for i, x in enumerate(state_names) if x == state]
    for j in indices:
    #seg = map.states[state_names.index(state)]
        seg = map.states[j]
        col = m.to_rgba(df[df.State==state].Attendance*1000).tolist()[0]
        poly = Polygon(seg, 
            facecolor=col,
            edgecolor=col)
        ax1.add_patch(poly)



cmap = plt.cm.get_cmap(name="OrRd")
norm = mpl.colors.BoundaryNorm(np.arange(0,650,100),cmap.N)
m = cm.ScalarMappable(norm=norm, cmap="OrRd")
m.set_array(np.arange(0,600,10))

divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(m,ax=ax2,cax=cax)

for state in df.State:
    indices = [i for i, x in enumerate(state_names) if x == state]
    for j in indices:
    #seg = map.states[state_names.index(state)]
        seg = map.states[j]
        col = m.to_rgba(df[df.State==state].PeoplePerVenue).tolist()[0]
        poly = Polygon(seg, 
            facecolor=col,
            edgecolor=col)
        ax2.add_patch(poly)


ax1.set_title("People attending church weekly (per 1000 people)",fontsize=16)
ax2.set_title("People per religious venue",fontsize=16)
