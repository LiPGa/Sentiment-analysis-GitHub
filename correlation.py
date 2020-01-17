# -*- coding: utf-8 -*-
import json
import csv
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os
import seaborn as sns
import matplotlib.font_manager as font_manager

csv.field_size_limit(500 * 1024 * 1024)
font = font_manager.FontProperties(weight='regular',size=28)

# node degree of wach developer in network (not weighted)
def get_everyone_degree(input_file): 
    res = {}
    df = pd.read_csv(input_file)
    for i in df['name1']:
        res[i] = res.get(i,0)+1
    for j in df['name2']:
        res[j] = res.get(j,0)+1
    # print ('No. of developers:', len(res.keys()))
    return res

def get_everyone_issues(issue_file):
    df = pd.read_csv(issue_file)
    res = {}
    for index,line in df.iterrows():
        name = line['name']
        res[name] = res.get(name,0)+1
    # print ('No. of developers:', len(res.keys()))
    return res

def get_everyone_comments(issue_file):
    res = {}
    df = pd.read_csv(issue_file)
    for index,line in df.iterrows():
        name = line['name']
        comment_number = len(line['label'])
        res[name] = res.get(name,0)+comment_number
    # print ('No. of developers:', len(res.keys()))
    return res

def get_everyone_score(polarity_file):
    csvfile = open(polarity_file, 'r',encoding='utf-8')
    reader = csv.reader(csvfile)
    is_threejs = 'threejs' in polarity_file
    res2 = {}
    res = {}
    for line in reader:
        if (line[0] == 'name'): continue;
        a1 = line[0]
        if is_threejs:
            polarity = line[-1][2:-2]
        else:
            polarity = line[-1]
        res[a1] = res.get(a1, []) + [polarity]

    for i, j in res.items():
        positive_cnt = j.count('positive')
        negative_cnt = j.count('negative')
        neutral_cnt = j.count('neutral')
        if positive_cnt==0 and negative_cnt ==0:
            res2[i] = 0.0
        else:
            res2[i] = round((negative_cnt+neutral_cnt)/len(j),3)
    return res2

def get_elements(input_file, polarity_file,issue_file, output_file0):  
    df = pd.read_csv(input_file)

    res1 = get_everyone_degree(input_file)
    res2 = get_everyone_score(polarity_file)

    with open(output_file0, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['name1', 'name2', \
                         "consist_cnt", "inconsist_cnt", 'pos_consist_cnt', 'neg_consist_cnt', \
                         "consist_rate", "inconsist_rate", 'pos_consist_rate', 'neg_consist_rate', \
                         'cmt_cnt', 'issue_cnt', 'score1', 'score2','degree1','degree2'])

        for index, row in df.iterrows():
            p1 = str(row['p1']) if type(row['p1'])!='str' else row['p1']
            p2 = str(row['p2']) if type(row['p2'])!='str' else row['p2']
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

            cmt_cnt = len(p1) + len(p2) # 合作的评论数量是两人的评论数量求和

            degree1 = res1[name1]
            degree2 = res1[name2]
            score1 = res2[name1]
            score2 = res2[name2]

            writer.writerow([name1] + [name2] \
                            + [consist_cnt] + [inconsist_cnt] + [pos_consist_cnt] + [neg_consist_cnt] \
                            + [consist_rate] + [inconsist_rate] + [pos_consist_rate] + [neg_consist_rate] \
                            + [cmt_cnt] + [1] + [score1] + [score2]+[degree1]+[degree2])

def group_collaborations_by_name(output_file0, output_file1):
    df = pd.read_csv(output_file0)
    df1 = df.groupby(['name1', 'name2'])[
        'consist_rate', 'inconsist_rate', 'pos_consist_rate', 'neg_consist_rate', 'score1', 'score2','degree1','degree2'].mean()
    df2 = df.groupby(['name1', 'name2'])[
        'consist_cnt', 'inconsist_cnt', 'pos_consist_cnt', 'neg_consist_cnt', 'issue_cnt', 'cmt_cnt'].sum()
    df3 = pd.merge(df1, df2, how='left', on=['name1', 'name2'])
    df3.to_csv(output_file1, mode='w', header=True)

def plot(a,b,fig_type,repo_name):
    nums = {'cmt':'2-1-1','issue':'2-1-2','score':'2-2','influence':'2-3'}
    labels = {'cmt':'No. of comments','issue':'No. of issues','score':'Overall sentiment score','influence':"Collaborators Position Dif."}
    num = nums[fig_type]
    labelname=labels[fig_type]
    # if (type == 'cmt'):
    #     num = '2-1-1'
    #     labelname = 'No. of comments'
    # if (type == 'issue'):
    #     num = '2-1-2'
    #     labelname = 'No. of issues'
    # if (type == 'score'):
    #     num = '2-2'
    #     labelname = 'Overall sentiment score'
    # if (type == 'influence'):
    #     num = '2-3'
    #     labelname = "Collaborators Position Dif."
    # else:
    #     num = 'test'
    #     labelname = "Collaborators Position Dif."
    
    corr, pval = stats.spearmanr(a,b)
    print ('{:<20}cor: {:<20}pval: {:<20}'.format(labelname,round(corr,3),round(pval,3) if round(pval,3)!=0.0 else pval ))

    if num=='2-1-1': return;
    if num=='2-1-2': return;

    z = np.polyfit(a, b, 1)
    p = np.poly1d(z)
    fig, ax = plt.subplots(figsize = (10,10))

    plt.grid(linestyle='--', alpha=1)
    plt.tick_params(labelsize=13) 
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xticks(np.arange(0.0,1.2,0.2))
    ax.set_yticks(np.arange(0.0,1.2,0.2))
    ax.plot(a,p(a),"black",linewidth=2,alpha = 1,label='Spearman Correlation Coefficient='+str(round(corr,3)))
    # sns.regplot(x=a, y=b)
    if repo_name == 'openra':
        ax.scatter(a,b,alpha=0.7)
    else:
        ax.scatter(a,b,alpha=0.5)

    ax.set_xlabel('Sentiment Consistency', fontproperties=font)
    ax.set_ylabel(labelname, fontproperties=font)

    plt.savefig('plots/'+num+'/' + repo_name +'.png',  box_inches='tight')
    plt.cla()

def correlation(outfile,repo_name):
    df = pd.read_csv(outfile)
    sentimentConsistency = []
    commentCounts = []
    issueCounts = []
    commentNumbers = []
    issueNumbers = []
    averageSentiment = []
    influenceDifference = []

    for index, line in df.iterrows():
        commentCounts.append(line['cmt_cnt'])
        issueCounts.append(line['issue_cnt'])

    averageCommentCount = int(np.mean(commentCounts))
    averageIssueCount = int(np.mean(issueCounts))
    print ("Average Comment Count =", averageCommentCount)
    print ("Average Issue Count =", averageIssueCount)

    for index, line in df.iterrows():
        degree1 = line['degree1']
        degree2 = line['degree2']
        commentNumber = line['cmt_cnt']
        issueNumber = line['issue_cnt']
        if commentNumber >= averageCommentCount:
        # if issueNumber > averageIssueCount:
            sentimentConsistency.append(line['consist_rate'])
            commentNumbers.append(commentNumber)
            issueNumbers.append(issueNumber)
            averageSentiment.append((line['score1']+line['score2'])/2)
            influenceDifference.append(float(abs(degree1 - degree2)) / max(degree1, degree2))

    print (len(sentimentConsistency), 'pairs of collaborative relationship')

    commentNumbers = pd.Series(commentNumbers)
    issueNumbers = pd.Series(issueNumbers)
    averageSentiment = pd.Series(averageSentiment)
    influenceDifference = pd.Series(influenceDifference)
    sentimentConsistency = pd.Series(sentimentConsistency)

    # test correlation between collaboration and position difference
    # a = issueNumbers
    # b = influenceDifference
    # corr, pval = stats.spearmanr(a,b)
    # print ('{:<40}cor: {:<20}pval: {:<20}'.format(repo_name,round(corr,3),round(pval,3) if round(pval,3)!=0.0 else pval ))
    # fig, ax = plt.subplots(figsize = (10,10))
    # ax.scatter(a,b,alpha=0.5)
    # plt.savefig('plots/test/'+repo_name+'.png')

    # plot(sentimentConsistency,commentNumbers,'cmt',repo_name)
    # plot(sentimentConsistency,issueNumbers,'issue',repo_name)
    plot(sentimentConsistency,averageSentiment,'score',repo_name)
    # plot(sentimentConsistency,influenceDifference,'influence',repo_name)

def plot_comment(repo_name,outfile):
    df = pd.read_csv(outfile)
    a = []
    b = []
    c = []
    print (len(df['cmt_cnt']))
    t1 = 0
    t2 = 50
    t3 = 100
    for index, line in df.iterrows():
        s = line['cmt_cnt']
        if s > t1 and s < t2:
            a.append(line['consist_rate'])
        if s >= t2 and s <= t3:
            b.append(line['consist_rate'])
        if s > t3:
            c.append(line['consist_rate'])
    a = pd.Series(a)
    b = pd.Series(b)
    c = pd.Series(c)
    plt.figure(figsize = (6,6))

    ax = plt.subplot(111)

    # ax.boxplot([a, b, c],widths=0.6,meanline=False,showmeans=True,notch=False,whis=1.5,
    #         flierprops = {'marker':'o','color':'#054E9F'},
    #         boxprops = {'color':'#054E9F','linewidth':1.7}, # 设置箱体属性，填充色和边框色
    #         meanprops = {'marker':'D','markerfacecolor':'green'}, # 设置均值点的属性，点的形状、填充色
    #         medianprops = {'linestyle':'-','color':'#054E9F','linewidth':1.5})
    ax.violinplot([a, b, c],widths=0.6, showmeans=True,showmedians=False)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['<'+str(t2), str(t2)+'-'+str(t3),'>'+str(t3)],fontproperties=font)
    ax.set_xlabel('Comment Count',fontproperties=font)
    ax.set_ylabel('Sentiment Consistency',fontproperties=font)

    plt.grid(axis='y', linestyle = '-.', alpha=1)
    plt.savefig('plots/2-1/'+repo_name+'-comment.png', bbox_inches='tight')


def plot_issue(repo_name,outfile):
    df = pd.read_csv(outfile)
    a = []
    b = []
    c = []
    t1 = 0
    t2 = 25
    t3 = 50
    for index, line in df.iterrows():
        s = line['issue_cnt']
        if s > t1 and s < t2:
            a.append(line['consist_rate'])
        if s >= t2 and s <= t3:
            b.append(line['consist_rate'])
        if s > t3:
            c.append(line['consist_rate'])
    a = pd.Series(a)
    b = pd.Series(b)
    c = pd.Series(c)

    plt.figure(figsize = (6,6))
    ax = plt.subplot(111)

    # ax.boxplot([a, b, c],widths=0.6,meanline=False,showmeans=True,notch=False,whis=1.5,
    #         flierprops = {'marker':'o','color':'#054E9F'},
    #         boxprops = {'color':'#054E9F','linewidth':1.7},
    #         meanprops = {'marker':'D','markerfacecolor':'green'}, 
    #         medianprops = {'linestyle':'-','color':'#054E9F','linewidth':1.5})
    ax.violinplot([a, b, c],widths=0.6, showmeans=True,showmedians=False)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['<'+str(t2), str(t2)+'-'+str(t3),'>'+str(t3)],fontproperties=font)
    ax.set_xlabel('Issue Count',fontproperties=font)
    ax.set_ylabel('Sentiment Consistency',fontproperties=font)
    plt.grid(axis='y', linestyle = '-.')
    plt.savefig('plots/2-1/'+repo_name+'-issue.png', bbox_inches='tight')


def main():
    repo_names = ['threejs','pandas','ipython','grpc','openra']

    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', choices=repo_names+['all'], default='all')
    parser.add_argument('--plot', choices=[True,False], default=True)

    if not os.path.exists('plots/2-1/'):
        os.makedirs('plots/2-1/')
    if not os.path.exists('plots/2-2/'):
        os.makedirs('plots/2-2/')
    if not os.path.exists('plots/2-3/'):
        os.makedirs('plots/2-3/')    
    if not os.path.exists('plots/test/'):
        os.makedirs('plots/test/')

    args = parser.parse_args()
    plot = args.plot
    if args.repo!='all': candidates = []+[args.repo];
    else: candidates = repo_names;

    for repo_name in candidates:
        pre_path = 'Dataset/'+repo_name+'/'
        if not os.path.exists(pre_path+'RQ2/'):
            os.makedirs(pre_path+'RQ2')
        polarity_file = 'Dataset/'+repo_name +'/'+repo_name+'_polarity.csv'
        issue_file = 'Dataset/'+repo_name+'/'+repo_name+'_issue.csv'
        input_file = pre_path + 'RQ2/query_result.csv' # generated by querying issue_file 
        output_file0 = pre_path +'RQ2/collaboration0.csv'
        output_file1 = pre_path +'RQ2/collaboration1.csv'

        # test correlation between degree and avg issue comment for individual
        # res1 = get_everyone_comments(issue_file)
        # res2 = get_everyone_issues(issue_file)
        # res3 = get_everyone_degree(input_file)
        # comment_list, issue_list, degree_list = [], [], []
        # avg_issue_comment_list = []
        # for n in res1.keys():
        #     avg_issue_comment = float(res1[n])/res2[n]
        #     avg_issue_comment_list.append(avg_issue_comment)
        #     degree_list.append(res2.get(n,0))
        # a = pd.Series(avg_issue_comment_list)
        # b = pd.Series(degree_list)
        # corr, pval = stats.spearmanr(a,b)
        # print ('{:<40}cor: {:<20}pval: {:<20}'.format(repo_name,round(corr,3),round(pval,3) if round(pval,3)!=0.0 else pval ))
    
        get_elements(input_file, polarity_file,issue_file,output_file0)
        group_collaborations_by_name(output_file0, output_file1)
        
        if plot==True: # generate violin and scatter figures
            # plot_comment(repo_name,output_file1)
            # plot_issue(repo_name,output_file1)  # times of collaboration 
            correlation(output_file1,repo_name) # sentiment & position difference


if __name__ == '__main__':
    main()

