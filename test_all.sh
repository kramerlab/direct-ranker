echo "Start to test all scripts"
echo "========================="

cd data
if [ -d "MSLR-TEST" ]; then
    echo "Test dir is already created"
else
    echo "Create Test dir ..."
    mkdir MSLR-TEST
    mkdir MSLR-TEST/Fold1
    mkdir MSLR-TEST/Fold2
    mkdir MSLR-TEST/Fold3
    mkdir MSLR-TEST/Fold4
    mkdir MSLR-TEST/Fold5
    head -n 1000 MSLR-WEB10K/Fold1/train.txt > MSLR-TEST/Fold1/train.txt
    head -n 1000 MSLR-WEB10K/Fold1/test.txt > MSLR-TEST/Fold1/test.txt
    head -n 1000 MSLR-WEB10K/Fold2/train.txt > MSLR-TEST/Fold2/train.txt
    head -n 1000 MSLR-WEB10K/Fold2/test.txt > MSLR-TEST/Fold2/test.txt
    head -n 1000 MSLR-WEB10K/Fold3/train.txt > MSLR-TEST/Fold3/train.txt
    head -n 1000 MSLR-WEB10K/Fold3/test.txt > MSLR-TEST/Fold3/test.txt
    head -n 1000 MSLR-WEB10K/Fold4/train.txt > MSLR-TEST/Fold4/train.txt
    head -n 1000 MSLR-WEB10K/Fold4/test.txt > MSLR-TEST/Fold4/test.txt
    head -n 1000 MSLR-WEB10K/Fold5/train.txt > MSLR-TEST/Fold5/train.txt
    head -n 1000 MSLR-WEB10K/Fold5/test.txt > MSLR-TEST/Fold5/test.txt
fi
cd -

echo "Start to test Fig. 2 ..."
sleep 1
cd fig2
ln -s ../Rankers .
python gen_fig2a.py --test 1
python gen_fig2b.py --test 1
python gen_fig2c.py --test 1
python gen_fig2d.py --test 1
echo "Test Done"
echo "Clean folder ..."
sleep 1
rm Rankers log
cd -

echo "Start to test Fig. 3 ..."
sleep 1
cd fig3
ln -s ../Rankers .
python training_label.py --test 1
python training_label_mslr.py --test 1 --path ../data/MSLR-TEST/
python training_size.py --test 1
python training_size_mslr.py --test 1 --path ../data/MSLR-TEST/
python plot_fig3.py
echo "Test Done"
echo "Clean folder ..."
sleep 1
rm -r mslr_data synth_data Rankers
cd -

echo "Start to test Fig. 4 ..."
sleep 1
cd fig4
ln -s ../Rankers .
python plot_sorted_list.py --test 1 --data ../data/MSLR-TEST/
echo "Test Done"
echo "Clean folder ..."
sleep 1
rm Rankers
cd -

echo "Start to test Table 1 ..."
sleep 1
cd table1
ln -s ../Rankers .
rm -r run_scripts
python create_runscripts.py --data ../data/MSLR-TEST --jarpath .. --resultsdir results_ranklib --datalabel MQ2007 --test 1 --model adarank,lambda
python create_runscripts.py --data ../data/MSLR-TEST --jarpath .. --resultsdir results_ranklib --datalabel MQ2008 --test 1 --model adarank,lambda
python create_runscripts.py --data ../data/MSLR-TEST --jarpath .. --resultsdir results_ranklib --datalabel MSLR-WEB10K --test 1 --model adarank,lambda
python gridsearch.py --path ../data/MSLR-TEST --model DirectRankerV1 --jobs 5 --data test
python gridsearch.py --path ../data/MSLR-TEST --model DirectRanker --jobs 5 --data test
python gridsearch.py --path ../data/MSLR-TEST --model RankeNet --jobs 5 --data test
python gridsearch.py --path ../data/MSLR-TEST --model ListNet --jobs 5 --data test
python get_results.python
echo "Test Done"
echo "Clean folder ..."
sleep 1
rm -r run_scripts Rankers
cd -

echo "Start to test Table 2 ..."
sleep 1
cd table2
ln -s ../Rankers .
mkdir .log
python ranknet_eval.py --path ../data/MSLR-TEST/ --activation sigmoid --epoch 1 --data test > .log/log_sig_sgd
python ranknet_eval.py --path ../data/MSLR-TEST/ --activation tanh --epoch 1 --data test > .log/log_tanh_sgd
python ranknet_eval.py --path ../data/MSLR-TEST/ --activation hard_sigmoid --epoch 1 --data test > .log/log_hard_sgd
python ranknet_eval.py --path ../data/MSLR-TEST/ --activation sigmoid --optimizer Adam --data test --epoch 1 > .log/log_sig_adam
python ranknet_eval.py --path ../data/MSLR-TEST/ --activation tanh --optimizer Adam --data test --epoch 1 > .log/log_tanh_adam
python ranknet_eval.py --path ../data/MSLR-TEST/ --activation hard_sigmoid --optimizer Adam --data test --epoch 1 > .log/log_hard_adam
python ranknet_eval.py --plot 1
echo "Test Done"
echo "Clean folder ..."
sleep 1
rm -r Rankers .log
cd -

echo "Start to test Table 4 ..."
sleep 1
cd table4
ln -s ../Rankers .
python timeana.py --model DirectRankerV1 --path ../data/MSLR-TEST
python timeana.py --model RankNet --path ../data/MSLR-TEST
echo "Test Done"
echo "Clean folder ..."
sleep 1
rm -r output Rankers
cd -
