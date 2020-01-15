# -*- coding: utf-8 -*-
from SentiCR import SentiCR
import pandas as pd
import argparse
import csv
import json
import os


def train_model(load_from_disk):
	train_file = 'dataset_3000.csv'
	train_df = pd.read_csv(train_file)[['Polarity', 'Text']]
	classifier_model = SentiCR(algo="GBT", training_data=train_df, load_from_disk=load_from_disk)
	return classifier_model

def sentiment_analysis(classifier_model, test_set):
	pred = classifier_model.get_sentiment_polarity_collection(test_set)
	return pred
  
def classify(load_from_disk):
	# 训练模型并分类文本 结果在tmp的polarity.csv里
	repo_names = ['threejs','openra', 'pandas','ipython','grpc']
	classifier_model = train_model(load_from_disk)
	print("Training complete.")

	for repo_name in repo_names:
		print("Classifying "+repo_name+" comments...")
		read_file = '../Dataset/'+repo_name+'/'+repo_name+'_comments.csv'
		out_file = '../Dataset/'+repo_name+'/'+repo_name+'_polarity.csv'
		test_df = pd.read_csv(read_file)
		test_set = test_df.text
		pred = sentiment_analysis(classifier_model, test_set)
		test_df['polarity'] = pred
		test_df.to_csv(out_file, index=None)

def get_all_comments():
	repo_names = ['threejs','openra', 'pandas','ipython','grpc']
	for repo_name in repo_names:
		path = '../issue-data/'+repo_name
		files= os.listdir(path)
		files.sort()
		comment_file = '../Dataset/'+repo_name+'/'+repo_name+'_comments.csv'
		writer=csv.writer(open(comment_file , 'w', encoding = 'utf-8'))
		writer.writerow(['name', 'issue_number', 'date','text'])

		for file in files:
			if file == '.DS_Store': continue;
			issue_number = file.split('.')[0]
			issue_file_path = os.path.join(path, file)
			issue_jsonObject = open(issue_file_path, 'r', encoding = 'utf-8').read()

			if repo_name=='threejs':
				for comment_data in json.loads(issue_jsonObject)['data']:
					actor_name = comment_data['user']['login']
					sentence = str(comment_data['body'])
					sentence = sentence.replace('\r', ' ').replace('\n',' ').replace('  ',' ')
					date = comment_data['created_at'][:10]
					writer.writerow([actor_name] + [issue_number] + [date] + [sentence])
			else:
				for comment_data in json.loads(issue_jsonObject):
					actor_name = comment_data['user']['login']
					sentence = str(comment_data['body'])
					sentence = sentence.replace('\r', ' ').replace('\n',' ').replace('  ',' ')
					date = comment_data['created_at'][:10]
					writer.writerow([actor_name] + [issue_number] + [date] + [sentence])

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--train', choices=[True, False], default=False)
	parser.add_argument('--prepare', choices=[True, False], default=False)
	args = parser.parse_args()
	train = args.train
	prepare = args.prepare
	load_from_disk = not train 
	if prepare==True:  
		get_all_comments()
	classify(load_from_disk)


if __name__ == '__main__':
    main()