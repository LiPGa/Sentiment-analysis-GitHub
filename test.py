import csv
import pandas as pd
csv.field_size_limit(500 * 1024 * 1024)

repo_name = 'ipython'
polarity_file = 'Dataset/'+repo_name +'/'+repo_name+'_polarity.csv'
df = pd.read_csv(polarity_file)
l  = list(df['polarity'])
# l  = [i[2:-2] for i in list(df['polarity'])]
pos = l.count('positive')
neu = l.count('neutral')
neg = l.count('negative')
print (neu/(pos+neu+neg))

repo_names = ['threejs','pandas','ipython','grpc','openra']

for repo in repo_names:
	path = 'Dataset/'+repo+'/RQ2/collaboration1.csv'
	df = pd.read_csv(path)
	print(df['consist_cnt'].sum()-df['pos_consist_cnt'].sum()-df['neg_consist_cnt'].sum(), df['pos_consist_cnt'].sum(), df['neg_consist_cnt'].sum())