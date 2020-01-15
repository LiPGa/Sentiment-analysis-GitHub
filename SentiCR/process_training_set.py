import pandas as pd
import csv

df = pd.read_csv('training_set_3000.csv')
df = df.drop(["issue_id"], axis=1)
dic = dict()
dic['Negative'] = '-1'
dic['Neutral'] = '0'
dic['Positive'] = '0'

df['Annotation'] = df['Annotation'].map(lambda x: dic[x])
print(df)
df.to_csv("SW_training_set.csv", index=None)