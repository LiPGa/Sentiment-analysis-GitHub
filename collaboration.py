# -*- coding: utf-8 -*-
import os
import csv
import json
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import ks_2samp
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

csv.field_size_limit(500 * 1024 * 1024)

def get_all_developers(filepath): # 返回[(developer_name, comment_number)]
	csvfile = open(filepath, 'r',encoding='utf-8')
	reader = csv.reader(csvfile)
	res = {}
	for line in reader:
		res[line[0]] = res.get(line[0],0) + 1

	developers = [i for i in res.items() if i[1] >= 5]
	# developers = [i for i in res.items()]

	developers = sorted(developers, key = lambda item:-item[1])
	print ("Total developer number: ",len(developers))
	return developers


def get_top_developer(developers):
	return developers[0][0]

def count_consistent_pairs(list1, list2):
	res = list1.count('positive')*list2.count('positive')
	res += list1.count('negative')*list2.count('negative')
	res += list1.count('neutral')*list2.count('neutral')
	return res

def data_process(filepath, cr_file, ncr_file):
	developers = get_all_developers(filepath)
	csvfile = open(filepath, 'r', encoding = 'utf-8')
	reader = csv.reader(csvfile)
	res = {} # 从comment data中得到的developer,comment_number,polarity

	for line in reader:
		if (line[0]=='name'): continue;
		developer_name = line[0]
		date = line[1]
		issue_num = line[2]
		polarity = line[-1]
		res[(developer_name,issue_num)] = res.get((developer_name,issue_num),[])+[polarity]

	writer1 = csv.writer(open(cr_file,'w'))
	writer1.writerow(['developer1','developer2','cr_consistent_rate','co_issue_cnt','cmt_cnt1','cmt_cnt2'])

	writer2 = csv.writer(open(ncr_file,'w'))
	writer2.writerow(['developer1','developer2','ncr_consistent_rate','co_issue_cnt','cmt_cnt1','cmt_cnt2'])
	
	developer_num = len(developers)

	for i in range(developer_num):
		if i%50==0: print(i,'/',developer_num);
		(developer1,comment_number1) = developers[i]
		issue_numbers = []
		all_polarity1 = [] # all the comment polarity of developer1 

		for a,b in res.keys():
			if a == developer1:
				issue_numbers.append(b)
				all_polarity1 += res[(a,b)]


		for j in range(i+1,developer_num):
			(developer2,comment_number2) = developers[j]
			flag = False

			collaborate_issue_cnt = 0
			collaborate_pair_cnt = 0
			total_pair_count = 0
			collaborate_consistent_pair_cnt = 0
			total_consistent_pair_cnt = 0

			all_polarity2 = []
			collaborate_consistent_rate = []

			for c,d in res.keys():
				if c == developer2:
					all_polarity2 += res[(c,d)]
					if d in issue_numbers: # a shared participated issue is found
						flag = True
						collaborate_issue_cnt += 1
						polarity1 = res[(developer1,d)]
						polarity2 = res[(developer2,d)]
						collaborate_pair_cnt_one_issue = len(polarity1)*len(polarity2)
						collaborate_consistent_pair_cnt_one_issue = count_consistent_pairs(polarity1, polarity2)
						collaborate_pair_cnt += collaborate_pair_cnt_one_issue
						collaborate_consistent_pair_cnt += collaborate_consistent_pair_cnt_one_issue
						collaborate_consistent_rate.append(float(collaborate_consistent_pair_cnt_one_issue) / collaborate_pair_cnt_one_issue)

			total_pair_count += len(all_polarity1)*len(all_polarity2)
			total_consistent_pair_cnt += count_consistent_pairs(all_polarity1, all_polarity2)

			if total_pair_count > collaborate_pair_cnt: # No co-participated issues found
				no_collaborate_rate = round(float(total_consistent_pair_cnt - collaborate_consistent_pair_cnt)/(total_pair_count-collaborate_pair_cnt),3) 
				writer2.writerow([developer1, developer2, no_collaborate_rate, collaborate_issue_cnt, len(all_polarity1), len(all_polarity2)])
			if flag == True: 
				writer1.writerow([developer1, developer2, round(np.mean(collaborate_consistent_rate),3), collaborate_issue_cnt, len(all_polarity1), len(all_polarity2)])


def t_test(cr_file, ncr_file):
	df1 = pd.read_csv(ncr_file)
	df2 = pd.read_csv(cr_file)
	x=list(df1['ncr_consistent_rate'].values)
	y = []
	for index, line in df2.iterrows():
		if line['co_issue_cnt'] > 3:
			y.append(line['cr_consistent_rate'])
	print(np.mean(x),np.mean(y))
	x = pd.Series(x)
	y = pd.Series(y)
	print (stats.levene(x, y))
	print (stats.ttest_ind(x,y, equal_var = False))

def plot_q1():
    writer = csv.writer(open('q1.csv', 'w'))
    writer.writerow(['repo', 'type', 'sentiment_consistency'])

    repos = ['grpc','threejs','ipython','openra','pandas']
    for repo_name in repos:
        co_file = 'Dataset/'+repo_name+'/q1/co_relation.csv'
        non_file = 'Dataset/'+repo_name+'/q1/non_relation.csv'
        df1 = pd.read_csv(co_file)
        df2 = pd.read_csv(non_file)
        for i,line in df1.iterrows():
            if line['cr_consistent_rate']>1: continue;
            writer.writerow([repo_name,'collaborative',line['cr_consistent_rate']])
        for i, line in df2.iterrows():
            if line['ncr_consistent_rate']>1: continue;
            writer.writerow([repo_name, 'non-collaborative', line['ncr_consistent_rate']])

def box_plot_one():
	plot_q1()
	df = pd.read_csv('q1.csv')
	plt.ylim(0,1)
	fig, axes = plt.subplots(figsize=(23, 10))
	sns.set_style("whitegrid")
	sns.boxplot(x = 'repo',y='sentiment_consistency',data=df, hue='type',orient='v',
				ax=axes,hue_order=['non-collaborative','collaborative'],palette="Set3",
				width = 0.8,linewidth=3,whis=1.5,order=['threejs','pandas','ipython','grpc','openra'])
	plt.xticks(fontproperties=font)
	plt.yticks(fontproperties=font)
	plt.legend(loc='lower right',fontsize=20)
	axes.set_xlabel('Repo name',fontproperties=font)
	axes.set_ylabel('Sentiment consistency rate',fontproperties=font)
	fig.savefig('plots/q1.png')

def merge_data_csv():
	writer = csv.writer(open('q1.csv', 'w'))
	writer.writerow(['repo', 'type', 'sentiment_consistency'])

	repos = ['grpc','threejs','ipython','openra','pandas']
	repo_names = ['gRPC', 'Three.js', 'IPython', 'OpenRA', 'Pandas']
	for i in range(len(repos)):
		repo_name = repos[i]
		name = repo_names[i]
		print("start", name)
		co_file = 'Dataset/'+repo_name+'/RQ1/'+repo_name+'_CR.csv'
		non_file = 'Dataset/'+repo_name+'/RQ1/'+repo_name+'_NCR.csv'

		df1 = pd.read_csv(co_file)
		for i,line in df1.iterrows():
			writer.writerow([name,'collaborator',round(line['cr_consistent_rate'],2)])
		df2 = pd.read_csv(non_file)
		for i, line in df2.iterrows():
			writer.writerow([name, 'non-collaborator', round(line['ncr_consistent_rate'],2)])

def box_plot_one(path):
	font = font_manager.FontProperties(weight='regular',size=22)
	df = pd.read_csv('q1.csv')
	plt.ylim(0,1)
	fig, axes = plt.subplots(figsize=(20, 7))
	sns.set_style("whitegrid")

	sns.boxplot(x = 'repo',y='sentiment_consistency',data=df, hue='type',orient='v',
				ax=axes,hue_order=['non-collaborator','collaborator'],palette="Set3",
				width = 0.75,linewidth=2,whis=1.5,order=['Three.js','Pandas','IPython','gRPC','OpenRA'])
	plt.xticks(fontproperties=font)
	plt.yticks(fontproperties=font)
	plt.legend(loc='lower right',fontsize=20)
	axes.set_xlabel('Repository',fontproperties=font)
	axes.set_ylabel('Sentiment consistency rate',fontproperties=font)
	plt.show()
	fig.savefig(path)

def main():
	repo_names = ['threejs','pandas','ipython','grpc','openra']

	parser = argparse.ArgumentParser()
	parser.add_argument('--repo', choices=repo_names, default='threejs')
	parser.add_argument('--plot', choices=[True,False], default=True)
	parser.add_argument('--path', default='plots/RQ1.png')

	args = parser.parse_args()
	repo_name = args.repo
	plot = args.plot

	pre_path = 'Dataset/'+repo_name+'/'
	filepath = pre_path +repo_name+'_polarity.csv'
	cr_file = pre_path +'RQ1/'+ repo_name+ '_CR.csv'
	ncr_file = pre_path +'RQ1/'+ repo_name+ '_NCR.csv'
	print ('Processing ',repo_name + '...')
	# data_process(filepath, cr_file, ncr_file)
	# t_test(cr_file, ncr_file)
	if plot:
		merge_data_csv()
		box_plot_one(path)


if __name__ == '__main__':
    main()