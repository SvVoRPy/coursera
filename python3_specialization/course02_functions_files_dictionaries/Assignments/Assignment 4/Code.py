import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
working_directory="C:/Users/SvenV/Desktop/Coursera Kurse/Python Spezialization II/Assignments/Assignment 4/"

data=pd.read_csv(working_directory+'Football_Teams.csv',sep=";")

# Percentage won:
data['won_perc']=data['wins']/(data['wins']+data['losses']+data['ties'])

# Teams:
data['team'].value_counts()

# Calculate Moving Averages
data['won_perc_ma']=data['won_perc'].rolling(center=False,window=10).mean()

# Source for RGB Colors:
# https://moncpc.files.wordpress.com/2010/02/nationalfootballleague_frc_2000_sol_srgb.pdf
cs=['#2C5E4F','#DD4814','#2A6EBB','#4B306A']

team=["Green Bay Packers","Chicago Bears","Detroit Lions","Minnesota Vikings"]

fig, ax = plt.subplots(1,1,figsize=(10,6)); 
for team,color in zip(team,cs):
    team_data = data[data.team==team]
    team_data.plot(x='season', y='won_perc_ma',
          kind='line',ax=ax,
          marker='o',markersize=5,
          color=color,lw=3,label=team)
plt.ylim(-0.02,0.8)
plt.yticks([0,0.25,0.5,0.75],['0 %','25 %','50 %','75 %'], alpha=0.8)
[plt.axhline(_x, linewidth=1, linestyle='--', color='grey') for _x in [0,0.25,0.5,0.75]]
plt.tick_params(top='off', bottom='off', labelbottom='on')
plt.title("Football NFC North Wins (%) 1970 to 2016 \n (10 Year Moving Average)")
plt.legend(["Green Bay Packers","Chicago Bears","Detroit Lions","Minnesota Vikings"],
           loc=8,ncol=2,frameon=False)
sns.set_style("whitegrid", {'axes.grid' : False})
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.xlabel("Season Played")
plt.ylabel("Wins as Percent of All Games")

plt.savefig(working_directory+'Assignment4.png',dpi=400)