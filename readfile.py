import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

# read the raw data
data = pd.read_csv('bank-additional/bank-additional-full.csv')

#list the column names
label = ['age','job',"marital","education","default","housing","loan","contact","month","day_of_week","duration","campaign","pdays","previous","poutcome","emp.var.rate","cons.price.idx","cons.conf.idx","euribor3m","nr.employed","y"]

#read lines in the raw data and split by semi-comma to get column values
for i in data:
    value = data[i][:].str.split(';')

for i in range(len(value)):
    for j in range(len(value[i])):
        if '"' in value[i][j]:
            value[i][j] = value[i][j][1:-1]

#loop the data to get arraged structure
temp = []
for index in range(len(value)):
    index = pd.DataFrame({label[i]: [value[index][i]] for i in range(len(label))})
    temp.append(index)

result = pd.concat(temp, ignore_index=True)

result.to_csv('original_data.csv')