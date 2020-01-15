# -*- coding: utf-8 -*-
import sys
import argparse
import json
import csv
import os
import argparse
import re
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def group_comments(polarity_file,issue_file):
	print("Processing issue...")
	f = open(issue_file, 'w',encoding='utf-8') 
	writer=csv.writer(f)
	writer.writerow(['name', 'issue_num', 'label'])
	df = pd.read_csv(polarity_file)
	lstg = df.groupby(['issue_number'])
	for key,group in lstg:
		res  = {}
		issue_number = key
		polarity_list = list(group[['polarity']].T.values[0])
		name_list = list(group[['name']].T.values[0])
		for i in range(len(polarity_list)):
			polarity = polarity_list[i][2:-2]
			name = name_list[i]
			if polarity == 'positive':
				label = '1'
			if polarity == 'negative':
				label = '0'
			if polarity == 'neutral':
				label = '2'
			res[name] = res.get(name,'') + label
		for k, v in res.items():
			writer.writerow([k] + [issue_number] + [v])

def group_comments_by_week(polarity_file, label_week_file):
	print("Processing week...")

	def get_week_number(date):
		datetime_object = datetime.strptime(date, '%Y-%m-%d')
		week_number = datetime_object.isocalendar()[1]
		year_week = str(date[:4]) + str(week_number)
		return year_week
	
	f = open(label_week_file, 'w',encoding='utf-8') 
	writer=csv.writer(f)
	writer.writerow(['name', 'year_week', 'issue_num', 'label'])
	
	df = pd.read_csv(polarity_file)
	lstg = df.groupby(['issue_number'])
	for key,group in lstg:
		res  = {}
		issue_number = key
		polarity_list = list(group[['polarity']].T.values[0])
		name_list = list(group[['name']].T.values[0])
		date_list = list(group[['date']].T.values[0])
		for i in range(len(polarity_list)):
			polarity = polarity_list[i][2:-2]
			name = name_list[i]
			date = date_list[i]
			if polarity == 'positive':
				label = '1'
			if polarity == 'negative':
				label = '0'
			if polarity == 'neutral':
				label = '2'
			year_week = str(get_week_number(date))

			res[name,year_week] = res.get((name,year_week),'') + label

		for k, v in res.items():
			writer.writerow([k[0]] + [k[1]] + [issue_number] + [v])

def group_comments_by_period(polarity_file, label_period_file):
	print("Processing period...")
	def string_toDatetime(string):
		return datetime.strptime(string, "%Y-%m-%d")
	f = open(label_period_file, 'w',encoding='utf-8') 
	writer=csv.writer(f)
	writer.writerow(['name', 'period', 'issue_num', 'label'])
	
	df = pd.read_csv(polarity_file)
	lstg = df.groupby(['issue_number'])
	for key,group in lstg:
		res  = {}
		issue_number = key
		polarity_list = list(group[['polarity']].T.values[0])
		name_list = list(group[['name']].T.values[0])
		date_list = list(group[['date']].T.values[0])
		for i in range(len(polarity_list)):
			polarity = polarity_list[i][2:-2]
			name = name_list[i]
			date = date_list[i]
			if polarity == 'positive':
				label = '1'
			if polarity == 'negative':
				label = '0'
			if polarity == 'neutral':
				label = '2'
			date = string_toDatetime(date)
			period = str(date.year) + str(int(not date<string_toDatetime(str(date.year)+'-06-30')))

			res[name,period] = res.get((name,period),'') + label

		for k, v in res.items():
			writer.writerow([k[0]] + [k[1]] + [issue_number] + [v])


def main():
	parser = argparse.ArgumentParser()
	repo_names = ['threejs','openra', 'pandas','ipython','grpc']
	parser.add_argument('--repo', choices=repo_names, default='threejs')
	args = parser.parse_args()
	repo_name = args.repo

	comment_file = 'Dataset/'+repo_name+'/'+repo_name+'_comments.csv'
	issue_file = 'Dataset/'+repo_name+'/'+repo_name+'_issue.csv'
	polarity_file = 'Dataset/'+repo_name+'/'+repo_name+'_polarity.csv'
	label_week_file = 'Dataset/'+repo_name+'/'+repo_name+'_label_week.csv'
	label_period_file = 'Dataset/'+repo_name+'/'+repo_name+'_label_period.csv'
	group_comments(polarity_file,issue_file)
	group_comments_by_week(polarity_file, label_week_file)
	group_comments_by_period(polarity_file, label_period_file)


if __name__ == '__main__':
    main()



