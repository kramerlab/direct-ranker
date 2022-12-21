# Loading data
For loading the data go into the folders `data` and run:

```bash
source get_data.sh
```
For MQ2007/8 we remove all the zeros in the queries with the script `delete_zeros.py`.

# Virtualenv

```bash
pip install virtualenv
virtualenv DirectRanker
source DirectRanker/bin/activate
pip install -r requirements.txt
```

# Test the code

```bash
./test_all.sh
```

# Figure 2
For generating the plots from Figure 2 one should run the following scripts in the folder `fig2`:

1. Generating the synthetic data:
```bash
python gen_fig2a.py  # for Fig. 2 a)
python gen_fig2b.py  # for Fig. 2 b)
python gen_fig2c.py  # for Fig. 2 c)
python gen_fig2d.py  # for Fig. 2 d)
```

# Figure 3
For generating the plots from Figure 3 one should run the following scripts in the folder `fig3`:

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
python plot_fig3.py
```

# Figure 4
For generating the plots from Figure 4 one should run the following scripts in the folder `fig4`:

1. Generating the synthetic data:
```bash
ln -s ../Rankers .
python plot_sorted_list.py
```

# Table 2
For generating table 2 one should run the following scripts in the folder `table2`:

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
python gridsearch.py --path ../data/MQ2007/ --model RankNet --jobs 5 --data MQ2007
python gridsearch.py --path ../data/MQ2008/ --model RankNet --jobs 5 --data MQ2008
python gridsearch.py --path ../data/MSLR-WEB10K/ --model RankNet --jobs 5 --data MSLR10K
python gridsearch.py --path ../data/MQ2007/ --model ListNet --jobs 5 --data MQ2007
python gridsearch.py --path ../data/MQ2008/ --model ListNet --jobs 5 --data MQ2008
python gridsearch.py --path ../data/MSLR-WEB10K/ --model ListNet --jobs 5 --data MSLR10K
python gridsearch.py --path ../data/MQ2007/ --model DirectRanker --jobs 5 --data MQ2007
python gridsearch.py --path ../data/MQ2008/ --model DirectRanker --jobs 5 --data MQ2008
python gridsearch.py --path ../data/MSLR-WEB10K/ --model DirectRanker --jobs 5 --data MSLR10K
```

4. Generating the Table / Friedman Test
```bash
python get_results.py
```

# Table 3
For generating table 3 one should run the following scripts in the folder `table3`:

```bash
./run_time_ana.sh
```
The different time results for RankLib are printed in the terminal while the results for the
DirectRanker and RankNet are stored in an output folder.

# Table 4
For generating table 4 one should run the following scripts in the folder `table4`:

```bash
./run_ranknet_eval.sh
python ranknet_eval.py --plot 1
```
