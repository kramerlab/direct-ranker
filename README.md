# DirectRanker

This is a Python implementation of the DirectRanker. It does not
include all tests should be suffiecient for an initial evaluation on
the LeToR data sets.

The script has been designed to be run on a recnet Ubuntu-based Linux
machine. It has also been tested on OSX. 


## Dependencies

The libraries required for the python script are as follows:
numpy
tensorflow
sklearn

Please install them using pip.


## Installation

First the LeToR data sets (MSLR-WEB10K, MQ2008, MQ2007) needs to be download form [here](https://www.microsoft.com/en-us/research/project/mslr/) 
and placed in the script as shown below



```python (sandbox.py)
...
x_train, y_train, q_train = readData(data_path="PATH_TO_LETOR/Fold1/train.txt", binary=True, at=10)
x_test, y_test, q_test = readData(data_path="PATH_TO_LETOR/Fold1/test.txt", binary=True, at=10)
...
``` 

Then just run the script with

```bash
python3.6 sandbox.py
```


it will then print out the NDCG@k set in

```python
...
nDCGScorer(dr, x_test, y_test, at=k)
...
```

With the scripts gridsearch_dataset.py you can perform a gridsearch on the LeToR datas.

