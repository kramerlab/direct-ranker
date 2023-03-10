mkdir .log
ln -s ../Rankers .

python ranknet_eval.py --path ../data/MSLR-WEB10K/ --activation sigmoid --epoch 10 > .log/log_sig_sgd &
python ranknet_eval.py --path ../data/MSLR-WEB10K/ --activation tanh --epoch 10 > .log/log_tanh_sgd &

python ranknet_eval.py --path ../data/MSLR-WEB10K/ --activation sigmoid --optimizer Adam --epoch 10 > .log/log_sig_adam &
python ranknet_eval.py --path ../data/MSLR-WEB10K/ --activation tanh --optimizer Adam --epoch 10 > .log/log_tanh_adam &
