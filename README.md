
# Sentiment analysis over collaborative relationships in OSS projects

Soure code and dataset.

```plain
.
├── README.md
├── process_data.py
├── collaboration.py
├── correlation.py
├── segment.py
├── analysis.py
├── network.py
├── network_plot.py
├── relationships.csv
├── issue-data/
├── Dataset/
    └── grpc/
	    └── grpc_comments.csv
	    └── grpc_polarity.csv
	    └── grpc_label_week.csv
	    └── grpc_label_period.csv
	    └── RQ1/
	    └── RQ2/
      	└── RQ3/
      	└── network/
    └── ipython/
    └── pandas/
    └── openra/
    └── threejs/
├── SentiCR/
    └── classifier.pkl
    └── vector.pkl
    ...
    └── sentiment.py
```

## Data set
The raw data sets crawled from GitHub are in `issue-data/`. 
- mrdoob/three.js
- pandas-dev/pandas
- ipython/ipython
- grpc/grpc
- OpenRA/OpenRA

| Project name | URL | Language | Gathered issues | No. of Developers | Star |
| ------ | ------ | ------ | ------ | ------ | ------ |
| mrdoob/three.js | https://github.com/mrdoob/three.js | JavaScript | 5465 | 2011 | 50249 |
| ipython/ipython | https://github.com/scala/scala | Python | 10172 | 3413 | 13478 |
| pandas-dev/pandas | https://github.com/pandas-dev/pandas | Python | 22854 | 5628 | 18832 |
| grpc/grpc | https://github.com/grpc/groc | C++ | 14828 | 3142 | 20560 |
| OpenRA/OpenRA | https://github.com/OpenRA/OpenRA | C# | 6026 | 682 | 6210 |

## Dependencies

- Python >= 3.6
- matplotlib == 3.1.1
- numpy == 1.18.0
- pandas == 0.24.2
- statsmodels == 0.10.2
- scipy == 1.1.0
- networkx == 2.1

## Usage
1. Sentiment analysis
classifier and vector models are saved in `classifier.pkl` and ` vector.pkl`. 
```python SentiCR/sentiment.py --train False --prepare False```
- prepare will decide whether to preprocess data for sentiment analysis. If true it will generate `($repo)_comments.csv` in repo folder from raw `issue-data/`.

2. Prepare data
```python process_data.py --repo $repo_name```
this will group comments by issue, by week and by half-a-year period seperately.

3. Collaboration difference (Q1)
```python collaboration.py --repo $repo_name ```
```python collaboration.py --plot True --path $figure_path ```
this will perform t-test on collaborative relationships vs. non-collaborative relationships, and generate boxplot.
if `plot` is true, this will merge relationships from 5 repos and generate a box plot.

4. Correlation analysis (Q2)
```python correlation.py --repo $repo_name、--plot True```
this will perform correlation analysis on 3 factors and generate violin graph and scatter graph.

5. Granger causality test (Q3/1)
```python segment.py --repo $repo_name```
```python analysis.py  --repo $repo_name --lag=2```
lag: number of weeks applied to causality test. You can change the number to 2/4/8/12.

6. Network approach (Q3/2)
```python network.py --repo $repo_name```
this will generate parameters of three network models
```python network_plot.py --param average_centrality --path $some_folder```
