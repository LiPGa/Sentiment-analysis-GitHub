# -*- coding: utf-8 -*-
import sys
import re
import os
import argparse
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime,date,time,timedelta
import csv
from dateutil.relativedelta import relativedelta
from datetime import datetime
from networkx import *
import community
plt.switch_backend('agg')

def datetime_toString(dt):
    return dt.strftime("%Y-%m-%d")

def string_toDatetime(string):
    return datetime.strptime(string, "%Y-%m-%d")


def add_issue_cnt(input_file, output_file0):
    df = pd.read_csv(input_file)
    df['p1'] = df['p1'].apply(lambda x: str(int(x)))
    df['p2'] = df['p2'].apply(lambda x: str(int(x)))

    with open(output_file0, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['name1', 'name2', 'period', \
                         "consist_cnt", "inconsist_cnt", 'pos_consist_cnt', 'neg_consist_cnt', \
                         "consist_rate", "inconsist_rate", 'pos_consist_rate', 'neg_consist_rate', \
                         'cmt_cnt', 'issue_cnt', 'score1', 'score2'])

        for index, row in df.iterrows():
            p1 = row['p1']
            p2 = row['p2']
            name1 = row['name1']
            name2 = row['name2']
            neg_consist_cnt = p1.count('0') * p2.count('0')
            pos_consist_cnt = p1.count('1') * p2.count('1')
            neutral_consist_cnt = p1.count('2') * p2.count('2')
            consist_cnt = neg_consist_cnt + pos_consist_cnt + neutral_consist_cnt
            inconsist_cnt = p1.count('0') * p2.count('1') + p1.count('1') * p2.count('0')
            pair_cnt = len(p1) * len(p2)
            consist_rate = round(float(consist_cnt) / pair_cnt, 2)
            inconsist_rate = round(float(inconsist_cnt) / pair_cnt, 2)

            pos_consist_rate = round(float(pos_consist_cnt) / pair_cnt, 2)
            neg_consist_rate = round(float(neg_consist_cnt) / pair_cnt, 2)
            # pos_consist_rate = round(float(pos_consist_cnt)/consist_cnt,2) if consist_cnt else 0.0
            # neg_consist_rate = round(float(neg_consist_cnt)/consist_cnt,2) if consist_cnt else 0.0

            cmt_cnt = len(p1) + len(p2)
            period = row['period']
            score1 = 1 - round(float(p1.count('0')) / len(p1), 2)
            score2 = 1 - round(float(p2.count('0')) / len(p2), 2)

            writer.writerow([name1] + [name2] + [period] \
                            + [consist_cnt] + [inconsist_cnt] + [pos_consist_cnt] + [neg_consist_cnt] \
                            + [consist_rate] + [inconsist_rate] + [pos_consist_rate] + [neg_consist_rate] \
                            + [cmt_cnt] + [1] + [score1] + [score2])


def group_collaborations_by_name_time(output_file0, output_file1):
    df = pd.read_csv(output_file0)
    df1 = df.groupby(['name1', 'name2', 'period'])[
        'consist_rate', 'inconsist_rate', 'pos_consist_rate', 'neg_consist_rate', 'score1', 'score2'].mean()
    df2 = df.groupby(['name1', 'name2', 'period'])[
        'consist_cnt', 'inconsist_cnt', 'pos_consist_cnt', 'neg_consist_cnt', 'issue_cnt', 'cmt_cnt'].sum()
    print(df1)
    df3 = pd.merge(df1, df2, how='left', on=['name1', 'name2', 'period'])
    df3.to_csv(output_file1, mode='w', header=True)



def separate_by_period(pre_path,output_file1):
	df = pd.read_csv(output_file1)
	df_gp = df.groupby(['period'])
	df_new = pd.DataFrame({'name1':[],'name2':[],'period':[],'consist_cnt':[]})
	for i, df_tmp in df_gp:
		df_tmp.drop(['consist_rate','inconsist_rate','pos_consist_rate',\
		             'neg_consist_rate'], axis=1, inplace=True)
		if not os.path.exists(pre_path + 'edge/'):
			os.makedirs(pre_path + 'edge/')
		edge_file = pre_path + 'edge/'+str(i) + '_edge.csv'
		if not os.path.exists(pre_path+'edge/'):
			os.makedirs(pre_path+'edge/')
		df_tmp.to_csv(edge_file, mode='w', header=True, index=None)


def param_to_csv(G,writer):
	try:
		communities = community.best_partition(G)
		global_modularity = round(community.modularity(communities, G),4)
		community_num = len(communities)
	except ValueError:
		return

	period = G.name[:5]
	date = string_toDatetime(period[:4] + '-01-01') +  relativedelta(months=6)*int(period[4])
	date = datetime_toString(date)
	node_num =  len(G)
	edge_num = number_of_edges(G)
	ave_degree = round(np.mean([a[1] for a in G.degree()]),3)
	dif_degree = round(np.std([a[1] for a in G.degree()]),3)

	edge_weights = [d['weight'] for (u,v,d) in G.edges(data=True)]
	ave_weight = round(np.mean(edge_weights),4)

	# diameter
	components = nx.connected_components(G)
	largest_component = max(components, key=len)
	subgraph = G.subgraph(largest_component)
	diameter = round(nx.diameter(subgraph),4)

	triadic_closure = round(nx.transitivity(G),4)
	density = round(nx.density(G),4)
	connectivity = round(float(edge_num)/node_num,2)
	# betweenness_dict = nx.betweenness_centrality(G)  # Run betweenness centrality
	# eigenvector_dict = nx.eigenvector_centrality(G)  # Run eigenvector centrality
	# nx.set_node_attributes(G, betweenness_dict, 'betweenness')
	# nx.set_node_attributes(G, eigenvector_dict, 'eigenvector')

	ave_diameter_l = []
	ave_clustering = average_clustering(G)
	for c in sorted(nx.connected_component_subgraphs(G),key=len,reverse=True):
		# print(type(c))
		ave_diameter_l.append(nx.diameter(c))
	ave_diameter = np.mean(ave_diameter_l)
	components_num =number_connected_components(G)
	close_centralitys =  [i[1] for i in closeness_centrality(G).items()]
	between_centralitys =  [i[1] for i in betweenness_centrality(G).items()]
	close_centrality = round(np.mean(close_centralitys),4)
	between_centrality = round(np.mean(between_centralitys),4)
	between_centrality_dif = round(np.var(between_centralitys),4)
	closeness_centrality_dif = round(np.std(close_centralitys),4)
	global_centrality = nx.global_reaching_centrality(G, normalized=True)
	writer.writerow([date, node_num, edge_num, ave_degree, dif_degree,ave_weight, components_num,\
					 diameter, ave_diameter, density, triadic_closure, global_modularity, community_num,\
					 close_centrality, between_centrality, between_centrality_dif, ave_clustering,connectivity,global_centrality])


def construct_param_csv(repo_name,pre_path,choice):
	print("Start constructing", choice, "network for", repo_name)
	files= os.listdir(pre_path+'edge/')
	if not os.path.exists(pre_path+'param/'):
		os.makedirs(pre_path+'param/')
	with open(pre_path + 'param/'+choice+'_param.csv', 'w') as f:
		writer = csv.writer(f)
		writer.writerow(['period','node_num','edge_num','ave_degree','dif_degree','ave_weight','components_num',\
						 'diameter','ave_diameter','density','triadic_closure','global_modularity','community_num',\
						 'closeness_centrality','between_centrality','between_centrality_dif','average_clustering','connectivity','global_centrality'])

		for file in sorted(files):
			period = file.split('_')[0]
			print (period)
			G1 = nx.Graph(name=period+'_'+choice)
			node_names = set()
			with open(pre_path+'edge/'+file, 'r') as edgecsv:
				edgereader = csv.reader(edgecsv)
				for e in edgereader:
					if e[0] == 'name1': continue; # first line
					if choice=='raw':
						if int(e[-1]) > 0: # number of comments
							G1.add_edge(e[0], e[1], weight=int(e[-1]))
							node_names.add(e[0])
							node_names.add(e[1])
					if choice =='consist':
						if int(e[5])>0:  # sentiment consistent
							G1.add_edge(e[0], e[1], weight=int(e[5]))
							node_names.add(e[0])
							node_names.add(e[1])
					if choice =='pos':
						if int(e[7])>0: 
							G1.add_edge(e[0], e[1], weight=int(e[7]))
							node_names.add(e[0])
							node_names.add(e[1])
					if choice =='neg':
						if int(e[8])>0: 
							G1.add_edge(e[0], e[1], weight=int(e[8]))
							node_names.add(e[0])
							node_names.add(e[1])
					if choice =='non-pos':
						G1.add_edge(e[0], e[1], weight=int(e[5])-int(e[7]))
						node_names.add(e[0])
						node_names.add(e[1])
					if choice =='non-neg':
						G1.add_edge(e[0], e[1], weight=int(e[5])-int(e[8]))
						node_names.add(e[0])
						node_names.add(e[1])

			G1.add_nodes_from(list(node_names))
			degree_dict = dict(G1.degree(G1.nodes()))
			nx.set_node_attributes(G1, degree_dict, 'degree')
			param_to_csv(G1,writer)



def main():
	repo_names = ['threejs','pandas','ipython','grpc','openra']

	parser = argparse.ArgumentParser()
	parser.add_argument('--repo', choices=repo_names, default='threejs')
	args = parser.parse_args()
	repo_name = args.repo

	pre_path = 'Dataset/'+repo_name+'/network/'
	input_file = pre_path + 'query_result.csv'
	output_file0 = pre_path + 'collaboration_0.csv'
	output_file1 = pre_path + 'collaboration_1.csv'

	add_issue_cnt(input_file, output_file0)
	group_collaborations_by_name_time(output_file0,output_file1)
	separate_by_period(pre_path,output_file1)

	# generate three network parameters
	construct_param_csv(repo_name, pre_path,'raw')
	construct_param_csv(repo_name, pre_path,'pos')
	construct_param_csv(repo_name, pre_path,'neg')


if __name__ == '__main__':
    main()