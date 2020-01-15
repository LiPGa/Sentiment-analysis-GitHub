# -*- coding: utf-8 -*-
import sys
import re
import argparse
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests
import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
global input_file, output_file
from scipy import stats
warnings.filterwarnings("ignore")

#计算AIC(k: number of variables, n: number of observations)
def AIC(y_test, y_pred, k, n):
    resid = y_test - y_pred
    SSR = sum(resid ** 2)
    AICValue = 2*k+n*log(float(SSR)/n)
    return AICValue

def test_stationary(ts):
	try:
		adfTest = adfuller(ts, autolag='AIC')
	except ValueError:
		return False

	if adfTest[1] > .05:
		return False
	return True

def granger_test_single(repo_name,choice,n):
	# print (os.path.dirname(__file__))
	pre_path = 'Dataset/'+repo_name+'/'
	relative_path = pre_path+'RQ3/single_collaboration/'   # 增加过缺失值的文件
	files= os.listdir(relative_path)

	total = 0
	cnt1 = 0
	cnt2 = 0
	cnt3 = 0
	cnt4 = 0
	cnt5 = 0
	cnt6 = 0
	cnt7 = 0
	cnt8 = 0


	for file in files:
		# print (total)
		if (file=='.DS_Store'): continue;
		if os.path.isfile(relative_path + file):
			df = pd.read_csv(relative_path + file)
		else: continue;

		if choice in range(1,5):
			df['cmt_cnt'] = [int(x) for x in df['cmt_cnt'].values]
			if (test_stationary(df['cmt_cnt'].values) == False): continue;

		if choice in range(5,9):
			df['score'] = df['score1']
			for i in range(len(df['score1'].values)):
				df['score'][i] = float(df['score1'][i]+df['score2'][i])/2
			if (test_stationary(df['score'].values) == False): continue;

		if choice%4 == 1:
			if (test_stationary(df['pos_consist_cnt'].values) == False): continue;
		if choice%4 == 2:
			if (test_stationary(df['neg_consist_cnt'].values) == False): continue;
		if choice%4 == 3:
			if (test_stationary(df['consist_cnt'].values) == False): continue;
		if choice%4 == 0:
			if (test_stationary(df['inconsist_cnt'].values) == False): continue;

		# x y  x是果 y是因。
		try:
			if choice == 1:
				my_dict1 = grangercausalitytests(df[['cmt_cnt','pos_consist_cnt']], maxlag=n, addconst=True, verbose=False)
			if choice == 2:
				my_dict2 = grangercausalitytests(df[['cmt_cnt','neg_consist_cnt']], maxlag=n, addconst=True, verbose=False)
			if choice == 3:
				my_dict3 = grangercausalitytests(df[['cmt_cnt','consist_cnt']], maxlag=n, addconst=True, verbose=False)
			if choice == 4:
				my_dict4 = grangercausalitytests(df[['cmt_cnt','inconsist_cnt']], maxlag=n, addconst=True, verbose=False)
			if choice == 5:
				my_dict5 = grangercausalitytests(df[['score','pos_consist_cnt']], maxlag=n, addconst=True, verbose=False)
			if choice == 6:
				my_dict6 = grangercausalitytests(df[['score','neg_consist_cnt']], maxlag=n, addconst=True, verbose=False)
			if choice == 7:
				my_dict7 = grangercausalitytests(df[['score','consist_cnt']], maxlag=n, addconst=True, verbose=False)
			if choice == 8:
				my_dict8 = grangercausalitytests(df[['score','inconsist_cnt']], maxlag=n, addconst=True, verbose=False)
			
		except ValueError: continue;

		total += 1
		if choice==1:
			for i in range(1,n+1):
				if my_dict1[i][0]['lrtest'][1] < 0.05:
					cnt1 += 1
					break
		if choice == 2:
			for i in range(1,n+1):
				if my_dict2[i][0]['lrtest'][1] < 0.05:
					cnt2 += 1
					break
		if choice==3:
			for i in range(1,n+1):
				if my_dict3[i][0]['lrtest'][1] < 0.05:
					cnt3 += 1
					break
		if choice ==4:
			for i in range(1,n+1):
				if my_dict4[i][0]['lrtest'][1] < 0.05:
					cnt4 += 1
					break
		if choice == 5:
			for i in range(1,n+1):
				if my_dict5[i][0]['lrtest'][1] < 0.05:
					cnt5 += 1
					break

		if choice==6:
			for i in range(1,n+1):
				if my_dict6[i][0]['lrtest'][1] < 0.05:
					cnt6 += 1
					break
		if choice==7:
			for i in range(1,n+1):
				if my_dict7[i][0]['lrtest'][1] < 0.05:
					cnt7 += 1
					break
		if choice==8:
			for i in range(1,n+1):
				if my_dict8[i][0]['lrtest'][1] < 0.05:
					cnt8 += 1
					break

	if choice == 1:
		print ("pos: {}/{}".format(cnt1,total))
		poss1.append(cnt1)
		poss2.append(total)
	if choice == 2:
		print ("neg: {}/{}".format(cnt2,total))
		negs1.append(cnt2)
		negs2.append(total)

	if choice == 3:
		print ("consist: {}/{}".format(cnt3,total))
		cons1.append(cnt3)
		cons2.append(total)

	if choice == 4:
		print ("inconsist: {}/{}".format(cnt4,total))
		incons1.append(cnt4)
		incons2.append(total)

	if choice == 5: print ("pos: {}/{}".format(cnt5,total));
	if choice == 6: print ("neg: {}/{}".format(cnt6,total));
	if choice == 7: print ("consist: {}/{}".format(cnt7,total));
	if choice == 8: print ("inconsist: {}/{}".format(cnt8,total));

def granger_new(repo_name, n):
	repo_names = ['threejs','pandas','ipython','grpc','openra']
	global poss1,poss2,negs1,negs2,cons1,cons2,incons2,incons1
	poss1 = []
	negs1 = []
	cons1 = []
	incons1 = []
	poss2 = []
	negs2 = []
	cons2 = []
	incons2 = []

	for repo_name in repo_names:
		print(repo_name, "lag =", n)
		granger_test_single(repo_name, 1,n)
		granger_test_single(repo_name, 2,n)
		granger_test_single(repo_name, 3,n)
		granger_test_single(repo_name, 4,n)
		print('\n')

	return sum(poss1),sum(poss2),sum(negs1),sum(negs2),sum(cons1),sum(cons2),sum(incons1),sum(incons2)


# test difference between two bernolli distributions
def test(cnt1,total1, cnt2, total2):
	l1 = np.zeros(total1)
	for i in range(cnt1):
		l1[i] = 1
	l2 = np.zeros(total2)
	for i in range(cnt2):
		l2[i] = 1
	l1 = pd.Series(l1)
	l2 = pd.Series(l2)
	print (stats.ttest_ind(l1,l2, equal_var = False))


def main():
	repo_names = ['threejs','pandas','ipython','grpc','openra']

	parser = argparse.ArgumentParser()
	parser.add_argument('--repo', choices=repo_names, default='threejs')
	parser.add_argument('--lag', default=2)

	args = parser.parse_args()
	repo_name = args.repo
	n = args.lag
	a1,a2,b1,b2,c1,c2,d1,d2 = granger_new(repo_name,n)
	
	print("significant relationship number:")
	print("positive:", a1,'/', a2)
	print("negative:",b1,'/',b2)
	print("consistency:",c1,'/',c2)
	print("inconsistency",d1,'/',d2)


if __name__ == '__main__':
    main()

