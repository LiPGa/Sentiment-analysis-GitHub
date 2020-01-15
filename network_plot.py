# -*- coding: utf-8 -*-
import sys
import re
import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime,date,time,timedelta
import csv
import argparse
from dateutil.relativedelta import relativedelta
from scipy.interpolate import interp1d
from scipy.interpolate import spline
import datetime 
import seaborn as sns
import matplotlib.font_manager as font_manager
import time
plt.switch_backend('agg')
sns.set()
sns.set_style("white")

def datetime_toString(dt):
    return dt.strftime("%Y-%m")
    # return dt.strftime("%Y-%m-%d")

def string_toDatetime(string):
    return datetime.strptime(string, "%Y-%m-%d")

def smooth(x,window_len=11,window='hanning'):
    if x.ndim != 1:
    	print("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
    	print("Input vector needs to be bigger than window size.")
    if window_len<3:
    	return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
    	print("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
    if window == 'flat': #moving average
   		w=np.ones(window_len,'d')
    else:  
    	w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='same')
    return y[window_len:-window_len+1]

def timestr_to_int(strlist):
	res = []
	for a in strlist:
		timeArray = time.strptime(a, "%Y-%m-%d %H:%M:%S")
		res.append(int(time.mktime(timeArray)))
	return np.array(res)

def int_to_time(intlist):
	res = []
	for a in intlist:
		dateArray = time.localtime(a)
		otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S",dateArray)
		res.append(otherStyleTime)
	return np.array(res)


def plot(repo_name,choice,plot_path):
	print ('\n'+repo_name)
	pre_path = 'Dataset/'+repo_name+'/network/'
	params_path = pre_path + 'param/'

	df1 = pd.read_csv(params_path+'pos_param.csv',index_col='period')
	df2 = pd.read_csv(params_path+'neg_param.csv',index_col='period')
	# df3 = pd.read_csv(params_path+'consist_param.csv',index_col='period')
	df4 = pd.read_csv(params_path+'raw_param.csv',index_col='period')
	# df5 = pd.read_csv(params_path+'non-pos_param.csv',index_col='period')
	# df6 = pd.read_csv(params_path+'non-neg_param.csv',index_col='period')
	
	df1.index = pd.to_datetime(df1.index)
	df2.index = pd.to_datetime(df2.index)
	# df3.index = pd.to_datetime(df3.index)
	df4.index = pd.to_datetime(df4.index)
	# df5.index = pd.to_datetime(df5.index)
	# df6.index = pd.to_datetime(df6.index)

	# choice = 'ave_diameter'
	ts1 = df1[choice]
	ts2 = df2[choice]
	# ts3 = df3[choice]
	ts4 = df4[choice]
	# ts5 = df5[choice]
	# ts6 = df6[choice]

	# ts1 = smooth(ts1)
	# ts2 = smooth(ts2)
	# ts3 = smooth(ts3)
	# ts4 = smooth(ts4)

	print ("positive",choice,round(np.mean(ts1),2))
	print ("negative",choice, round(np.mean(ts2),2))
	# print ("consistent",choice, round(np.mean(ts3),2))
	print ("original",choice, round(np.mean(ts4),2))
	# print ("non-positive average",choice, round(np.mean(ts5),2))
	# print ("non-negative ",choice, round(np.mean(ts6),2))
	# print (round(np.mean(ts3),2))

	y_label = choice
	if choice == 'clossness_centrality':
		choice = 'closeness_centrality'
	if choice =='average_clustering':
		y_label = 'mean clustering coefficient'
	if choice =='global_modularity':
		y_label = 'modularity'	
	if choice.startswith('between_centrality'):
		y_label = 'betweenness centrality'


	fig,ax = plt.subplots(figsize=(5,3))
	# plt.title('Time series of developer network - '+repo_name)
	plt.xlabel('date')
	plt.ylabel(choice)
	plt.ylim([-0.05, 1.05])
	plt.yticks(np.arange(0.0,1.2,0.2))

	tss = [ts1,ts2,ts4]
	colors = ['g','r','b','orange','r','g']
	# labels = ['pos','neg','consist','raw','non-pos','non-neg']
	labels = ['positive','negative','original']
	markers = ['s','o','^']
	
	# plt.subplot(211)

	for i in [0,1,2]:
		ts = tss[i]
		# ts = ts.resample('m')
		# ts = ts.interpolate(method='cubic')
		# ts.plot(color=colors[i],marker=markers[i],markersize=2.3, linewidth=1.5, label = labels[i])
		ts.plot(color=colors[i],marker=markers[i],markersize=1,linewidth=1.3, label = labels[i])

	plt.xlabel('')
	# ax.set_ylabel(choice,fontproperties=font)

	plt.ylabel(y_label,fontsize=17)
	plt.grid(linestyle='-', alpha=0.6, axis='y')
	# plt.legend(fontsize=8)
	if not os.path.exists(plot_path+choice+'/'):
		os.makedirs(plot_path+choice+'/')
	plt.savefig(plot_path+choice+'/'+repo_name+'.png')
	plt.clf()

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--param', choices=['global_modularity','between_centrality','average_clustering'], default='average_clustering')
	parser.add_argument('--path', default ='plots/4/')
	if not os.path.exists('plots/4'):
		os.makedirs('plots/4')
	args = parser.parse_args()
	choice = args.param
	path = args.path

	plot('threejs',choice,path)
	plot('pandas',choice,path)
	plot('ipython',choice,path)
	plot('grpc',choice,path)
	plot('openra',choice,path)


if __name__ == '__main__':
    main()





