# Loading data
For loading the data go into the folders `data` and run:

```bash
source get_data.sh
```
For MQ2007/8 we remove all the zeros in the queries with the script `delete_zeros.py`.

# Virtualenv & Compile AdaRank

```bash
# install all the python requirements
pip install virtualenv
virtualenv DirectRanker
source DirectRanker/bin/activate
pip install -r /path/to/requirements.txt

# install the Rust libary for AdaRank
# first install Rust: https://www.rust-lang.org/learn/get-started
pip install maturin
# install it in the virtualenv
cd rustlib
maturin develop -r
# install it in your global python env
maturin build
pip install target/wheels/*.whl
```

# Test the code

```bash
./test_all.sh
```

# Figure 4
For generating the plots from Figure 2 one should run the following scripts in the folder `fig4`:

1. Generating the synthetic data:
```bash
python gen_fig4a.py  # for Fig. 4 a)
python gen_fig4b.py  # for Fig. 4 b)
python gen_fig4c.py  # for Fig. 4 c)
python gen_fig4d.py  # for Fig. 4 d)
```

# Figure 5
For generating the plots from Figure 5 one should run the following scripts in the folder `fig5`:

1. Run the model tests for the synthetic data:
```bash
python training_label.py
python training_size.py
```

2. Run for the MSLR-WEB10K data:
```bash
python training_label_mslr.py
python training_size_mslr.py
```

3. Run the plot script:
```bash
python plot_fig5.py
```

# Figure 6
For generating the plots from Figure 6 one should run the following scripts in the folder `fig6`:

1. Generating the synthetic data:
```bash
ln -s ../Rankers .
python plot_sorted_list.py
```

# Table 1 and Figure 2/3
For generating table 1 one should run the following scripts in the folder `table1`:

1. Running the ranklib models
```bash
python create_runscripts.py --data ../data/MQ2007 --jarpath .. --resultsdir results_ranklib --datalabel MQ2007 # create run scripts for ranklib and MQ2007
python create_runscripts.py --data ../data/MQ2008 --jarpath .. --resultsdir results_ranklib --datalabel MQ2008 # create run scripts for ranklib and MQ2008
python create_runscripts.py --data ../data/MSLR-WEB10K --jarpath .. --resultsdir results_ranklib --datalabel MSLR-WEB10K # create run scripts for ranklib and MSLR-WEB10K
python run_jobs.py # run the ranklib models
```

2. Running the DirectRankerV1 model
```bash
ln -s ../Rankers .
python gridsearch.py --path ../data/MQ2007/ --model DirectRankerV1 --jobs 5 --data MQ2007
python gridsearch.py --path ../data/MQ2008/ --model DirectRankerV1 --jobs 5 --data MQ2008
python gridsearch.py --path ../data/MSLR-WEB10K/ --model DirectRankerV1 --jobs 5 --data MSLR10K
```

3. Running the tensorflow V2 models
```bash
python gridsearch.py --path ../data/MQ2007/ --model RankNet --jobs 5 --data MQ2007 --ttest 1 --binary 0
python gridsearch.py --path ../data/MQ2007/ --model RankNet --jobs 5 --data MQ2007 --ttest 1 --binary 1
python gridsearch.py --path ../data/MQ2008/ --model RankNet --jobs 5 --data MQ2008 --ttest 1 --binary 0
python gridsearch.py --path ../data/MQ2008/ --model RankNet --jobs 5 --data MQ2008 --ttest 1 --binary 1
python gridsearch.py --path ../data/MSLR-WEB10K/ --model RankNet --jobs 5 --data MSLR10K --ttest 1 --binary 0
python gridsearch.py --path ../data/MSLR-WEB10K/ --model RankNet --jobs 5 --data MSLR10K --ttest 1 --binary 1

python gridsearch.py --path ../data/MQ2007/ --model ListNet --jobs 5 --data MQ2007 --ttest 1 --binary 0
python gridsearch.py --path ../data/MQ2007/ --model ListNetOri --jobs 5 --data MQ2007 --ttest 1 --binary 0
python gridsearch.py --path ../data/MQ2007/ --model ListNet --jobs 5 --data MQ2007 --ttest 1 --binary 1
python gridsearch.py --path ../data/MQ2007/ --model ListNetOri --jobs 5 --data MQ2007 --ttest 1 --binary 1
python gridsearch.py --path ../data/MQ2008/ --model ListNet --jobs 5 --data MQ2008 --ttest 1 --binary 0
python gridsearch.py --path ../data/MQ2008/ --model ListNetOri --jobs 5 --data MQ2008 --ttest 1 --binary 0
python gridsearch.py --path ../data/MQ2008/ --model ListNet --jobs 5 --data MQ2008 --ttest 1 --binary 1
python gridsearch.py --path ../data/MQ2008/ --model ListNetOri --jobs 5 --data MQ2008 --ttest 1 --binary 1
python gridsearch.py --path ../data/MSLR-WEB10K/ --model ListNet --jobs 5 --data MSLR10K --ttest 1 --binary 0
python gridsearch.py --path ../data/MSLR-WEB10K/ --model ListNetOri --jobs 5 --data MSLR10K --ttest 1 --binary 0
python gridsearch.py --path ../data/MSLR-WEB10K/ --model ListNet --jobs 5 --data MSLR10K --ttest 1 --binary 1
python gridsearch.py --path ../data/MSLR-WEB10K/ --model ListNetOri --jobs 5 --data MSLR10K --ttest 1 --binary 1

python gridsearch.py --path ../data/MQ2007/ --model DirectRanker --jobs 5 --data MQ2007 --ttest 1 --binary 0
python gridsearch.py --path ../data/MQ2007/ --model DirectRanker --jobs 5 --data MQ2007 --ttest 1 --binary 1
python gridsearch.py --path ../data/MQ2008/ --model DirectRanker --jobs 5 --data MQ2008 --ttest 1 --binary 0
python gridsearch.py --path ../data/MQ2008/ --model DirectRanker --jobs 5 --data MQ2008 --ttest 1 --binary 1
python gridsearch.py --path ../data/MSLR-WEB10K/ --model DirectRanker --jobs 5 --data MSLR10K --ttest 1 --binary 0
python gridsearch.py --path ../data/MSLR-WEB10K/ --model DirectRanker --jobs 5 --data MSLR10K --ttest 1 --binary 1

python gridsearch.py --path ../data/MQ2007/ --model AdaRank --jobs 1 --data MQ2007 --ttest 1 --binary 0
python gridsearch.py --path ../data/MQ2007/ --model AdaRank --jobs 1 --data MQ2007 --ttest 1 --binary 1
python gridsearch.py --path ../data/MQ2008/ --model AdaRank --jobs 1 --data MQ2008 --ttest 1 --binary 0
python gridsearch.py --path ../data/MQ2008/ --model AdaRank --jobs 1 --data MQ2008 --ttest 1 --binary 1
python gridsearch.py --path ../data/MSLR-WEB10K/ --model AdaRank --jobs 1 --data MSLR10K --ttest 1 --binary 0
python gridsearch.py --path ../data/MSLR-WEB10K/ --model AdaRank --jobs 1 --data MSLR10K --ttest 1 --binary 1

python gridsearch.py --path ../data/MQ2007/ --model LambdaMart --jobs 5 --data MQ2007 --ttest 1 --binary 0
python gridsearch.py --path ../data/MQ2007/ --model LambdaMart --jobs 5 --data MQ2007 --ttest 1 --binary 1
python gridsearch.py --path ../data/MQ2008/ --model LambdaMart --jobs 5 --data MQ2008 --ttest 1 --binary 0
python gridsearch.py --path ../data/MQ2008/ --model LambdaMart --jobs 5 --data MQ2008 --ttest 1 --binary 1
python gridsearch.py --path ../data/MSLR-WEB10K/ --model LambdaMart --jobs 5 --data MSLR10K --ttest 1 --binary 0
python gridsearch.py --path ../data/MSLR-WEB10K/ --model LambdaMart --jobs 5 --data MSLR10K --ttest 1 --binary 1
```

4. Generating the Table / Friedman Test
```bash
python get_results.py
```

# Table 2
For generating table 2 one should run the following scripts in the folder `table2`:

```bash
./run_ranknet_eval.sh
python ranknet_eval.py --plot 1
```

# Table B1
For generating table B1 one should run the following scripts in the folder `tableB1`:

```bash
./run_time_ana.sh
```
The different time results for RankLib are printed in the terminal while the results for the
DirectRanker and RankNet are stored in an output folder.
