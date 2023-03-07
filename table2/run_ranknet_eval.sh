mkdir .log
ln -s ../Rankers .

python ranknet_eval.py --path ../data/MSLR-WEB10K/ --activation sigmoid --epoch 10 > .log/log_sig_sgd &
python ranknet_eval.py --path ../data/MSLR-WEB10K/ --activation tanh --epoch 10 > .log/log_tanh_sgd &
python ranknet_eval.py --path ../data/MSLR-WEB10K/ --activation hard_sigmoid --epoch 10 > .log/log_hard_sgd &
python ranknet_eval.py --path ../data/MSLR-WEB10K/ --activation linear --epoch 10 > .log/log_linear_sgd &

python ranknet_eval.py --path ../data/MSLR-WEB10K/ --activation sigmoid --optimizer Adam --epoch 10 > .log/log_sig_adam &
python ranknet_eval.py --path ../data/MSLR-WEB10K/ --activation tanh --optimizer Adam --epoch 10 > .log/log_tanh_adam &
python ranknet_eval.py --path ../data/MSLR-WEB10K/ --activation hard_sigmoid --optimizer Adam --epoch 10 > .log/log_hard_adam &
python ranknet_eval.py --path ../data/MSLR-WEB10K/ --activation linear --optimizer Adam --epoch 10 > .log/log_linear_adam &
