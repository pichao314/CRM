
from matplotlib import pyplot as plt
import seaborn as sns


#plot the corrlation map
plt.figure(figsize=(20, 20)) 
corr = data.corr()
h1 = sns.heatmap(corr, square=True, linewidths=.5, annot=True, fmt='.2f', cmap='coolwarm')

#plot the pie chart
df1 = data.y.value_counts().plot(figsize = (5,5),kind='pie',legend= True,title='CRM Term Deposit',fontsize=15,autopct='%.2f')