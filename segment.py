# -*- coding: utf-8 -*-
# reload(sys)
# sys.setdefaultencoding('utf-8')
import pandas as pd
import numpy as np
import argparse
import os
import time
from datetime import datetime,date,time,timedelta
import csv
from datetime import datetime
from dateutil.relativedelta import relativedelta


def datetime_toString(dt):
    return dt.strftime("%Y-%m-%d")

def string_toDatetime(string):
    return datetime.strptime(string, "%Y-%m-%d")

def to_first_weekday(year_week):
    year = str(year_week)[:4]
    week = int(str(year_week)[4:])
    tmp =  string_toDatetime(year + '-01-01') + relativedelta(weeks=week - 1)
    weekday = int(tmp.weekday())
    if weekday!=0:
        tmp = tmp + timedelta(days=-weekday)
    return datetime_toString(tmp)


def data_Full(df1):
    df1_date = df1['date'].tolist() 
    date_start = df1_date[0]

    date0 = string_toDatetime(date_start)
    date_s = date_start

    for j in range(0, len(df1_date)):
        date_i = df1_date[j]  
        while date_i != date_s:  
            adda = {'date': [date_s], 'consist_cnt': [0], 'inconsist_cnt': [0], \
                    'pos_consist_cnt': [0], 'neg_consist_cnt': [0], \
                    'issue_cnt': [0], 'cmt_cnt': [0], 'score1': [0], 'score2': [0]}
            date_da = pd.DataFrame(adda)
            # print('inserting', date_s)
            df1 = pd.concat([df1, date_da], ignore_index=True, sort=True)  
            date0 += relativedelta(weeks=1)
            date_s = datetime_toString(date0)  
        date0 += relativedelta(weeks=1) 
        date_s = datetime_toString(date0) 
    df1 = df1.sort_values(by=['date'])
    return df1


def get_week_number(time):
    d = time[:10]
    datetime_object = datetime.strptime(d, '%Y-%m-%d')
    return datetime_object.isocalendar()[1]


# Add issue_cnt; 
# `p1,p2` -> `consit_cnt inconsist_cnt s1,s2,pos_consist_cnt,neg_consist_cnt`;
# `year_week` -> `first weekday`
def add_issue_cnt(input_file, output_file0):
    df = pd.read_csv(input_file)
    df['p1'] = df['p1'].apply(lambda x: str(int(x)))
    df['p2'] = df['p2'].apply(lambda x: str(int(x)))

    with open(output_file0, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['name1', 'name2', 'date', \
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
            score1 = 1 - round(float(p1.count('0')) / len(p1), 2)
            score2 = 1 - round(float(p2.count('0')) / len(p2), 2)
            # date = row['date']
            date = to_first_weekday(row['year_week'])

            writer.writerow([name1] + [name2] + [date] \
                            + [consist_cnt] + [inconsist_cnt] + [pos_consist_cnt] + [neg_consist_cnt] \
                            + [consist_rate] + [inconsist_rate] + [pos_consist_rate] + [neg_consist_rate] \
                            + [cmt_cnt] + [1] + [score1] + [score2])


def group_collaborations_by_name_time(output_file0, output_file1):
    df = pd.read_csv(output_file0)
    df1 = df.groupby(['name1', 'name2', 'date'])[
        'consist_rate', 'inconsist_rate', 'pos_consist_rate', 'neg_consist_rate', 'score1', 'score2'].mean()
    df2 = df.groupby(['name1', 'name2', 'date'])[
        'consist_cnt', 'inconsist_cnt', 'pos_consist_cnt', 'neg_consist_cnt', 'issue_cnt', 'cmt_cnt'].sum()
    df3 = pd.merge(df1, df2, how='left', on=['name1', 'name2', 'date'])
    df3.to_csv(output_file1, mode='w', header=True)

def trans_week_to_date():
    df = pd.read_csv(output_file1)
    df['date'] = [to_first_weekday(i) for i in df['year_week'].values]
    df.to_csv(output_file1,index=None)


def get_all_density(pre_path,output_file1):
    df = pd.read_csv(output_file1)
    df_gp = df.groupby(['name1', 'name2'])
    df_new = pd.DataFrame({'name':[],'density':[],'length':[]})

    for i, df_tmp in df_gp:
        length = len(df_tmp)
        if (length<5): continue;
        start = df_tmp['date'].values[0] 
        end = df_tmp['date'].values[-1] 

        if (start==end): continue;

        name1 = i[0]
        name2 = i[1]

        delta = (string_toDatetime(end) - string_toDatetime(start)).days/7
        # density = float(length)/(12*delta.years+delta.months)
        density = float(length)/(delta)

        data = {'name':name1+'_'+name2,'density':[density],'length':[length]}
        df_new = pd.concat([df_new, pd.DataFrame(data)], ignore_index=True)
    df_new.to_csv(pre_path+'RQ3/density.csv',index=None)

def get_top_list(pre_path):
    top = 300
    df = pd.read_csv(pre_path+'RQ3/density.csv')
    df = df.sort_values(by = 'density', ascending=False)
    df = df.sort_values(by = 'length', ascending=False)
    return (df[:top])


def separate_collaboration(pre_path, output_file1): 
    if not os.path.exists(pre_path + 'RQ3/single_collaboration/'):
        os.makedirs(pre_path + 'RQ3/single_collaboration/')

    df = pd.read_csv(output_file1)
    df_gp = df.groupby(['name1', 'name2'])

    df_top = get_top_list(pre_path)
    cnt = 1

    for i, df_tmp in df_gp:
        file_name = i[0]+'_'+i[1]
        if (file_name not in df_top['name'].values):continue;
        relative_path = file_name + '.csv'
        print (cnt)
        cnt += 1
        path = pre_path + 'RQ3/single_collaboration/' + relative_path
        df_tmp.drop(['name1', 'name2', 'consist_rate','inconsist_rate','pos_consist_rate',\
                     'neg_consist_rate'], axis=1, inplace=True)
        df_tmp.sort_values(by="date", ascending=True)
        df_tmp = data_Full(df_tmp)
        df_tmp.to_csv(path, mode='w', header=True, index=None)


def main():
    repo_names = ['threejs','pandas','ipython','grpc','openra']

    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', choices=repo_names, default='threejs')

    args = parser.parse_args()
    repo_name = args.repo
    pre_path = 'Dataset/'+repo_name+'/'
    if not os.path.exists(pre_path+'RQ3/'):
        os.makedirs(pre_path+'/RQ3')

    input_file = pre_path + 'RQ3/query_result.csv'
    output_file0 = pre_path + 'RQ3/collaboration_0.csv'
    output_file1 = pre_path + 'RQ3/collaboration_1.csv'

    add_issue_cnt(input_file, output_file0)
    group_collaborations_by_name_time(output_file0, output_file1)
    get_all_density(pre_path,output_file1)
    separate_collaboration(pre_path,output_file1)


if __name__ == '__main__':
    main()