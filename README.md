# DyConAD
## Running the experiments
### Requirements:
The Python environment that we use  
- python >= 3.7
- Dependencies
```
pandas==1.5.3
numpy==1.23.4
Dgl-cu111==0.6.1
torch==1.9.0+cu111
```
### Dataset and preprocessing
#### Download the public data
Download the sample datasets from here and store their csv files in a folder named `<preprocess/data/>` .
- [Wikipedia](http://snap.stanford.edu/jodie/wikipedia.csv)
- [Reddit](http://snap.stanford.edu/jodie/reddit.csv)
- [ucr](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)

#### Model Training
```
python DyConAD_main.py -d wikipedia
```
