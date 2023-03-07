ln -s ../Rankers .

python timeana.py --model DirectRankerV1 --path ../data/MSLR-WEB10K
python timeana.py --model RankNet --path ../data/MSLR-WEB10K
python timeana.py --model ListNet --path ../data/MSLR-WEB10K
python timeana.py --model LambdaMart --path ../data/MSLR-WEB10K
python timeana.py --model AdaRank --path ../data/MSLR-WEB10K

mkdir .log

echo Fold1

(date +%s) > .log/start_time
java -jar ../RankLib-2.10.jar -train ../data/MSLR-WEB10K/Fold1/train.txt -ranker 0 -metric2t NDCG@10 -silent
(date +%s) > .log/cur_time
python -c "print(int(open(\".log/cur_time\").read()) - int(open(\".log/start_time\").read()))"

(date +%s) > .log/start_time
java -jar ../RankLib-2.10.jar -train ../data/MSLR-WEB10K/Fold1/train.txt -ranker 1 -metric2t NDCG@10 -silent
(date +%s) > .log/cur_time
python -c "print(int(open(\".log/cur_time\").read()) - int(open(\".log/start_time\").read()))"

(date +%s) > .log/start_time
java -jar ../RankLib-2.10.jar -train ../data/MSLR-WEB10K/Fold1/train.txt -ranker 2 -metric2t NDCG@10 -silent
ELAPSED_TIME=$(((date +%s) - $START_TIME))
(date +%s) > .log/cur_time
python -c "print(int(open(\".log/cur_time\").read()) - int(open(\".log/start_time\").read()))"

(date +%s) > .log/start_time
java -jar ../RankLib-2.10.jar -train ../data/MSLR-WEB10K/Fold1/train.txt -ranker 3 -metric2t NDCG@10 -silent
ELAPSED_TIME=$(((date +%s) - $START_TIME))
(date +%s) > .log/cur_time
python -c "print(int(open(\".log/cur_time\").read()) - int(open(\".log/start_time\").read()))"

(date +%s) > .log/start_time
java -jar ../RankLib-2.10.jar -train ../data/MSLR-WEB10K/Fold1/train.txt -ranker 3 -metric2t NDCG@10 -silent
ELAPSED_TIME=$(((date +%s) - $START_TIME))
(date +%s) > .log/cur_time
python -c "print(int(open(\".log/cur_time\").read()) - int(open(\".log/start_time\").read()))"

(date +%s) > .log/start_time
java -jar ../RankLib-2.10.jar -train ../data/MSLR-WEB10K/Fold1/train.txt -ranker 6 -metric2t NDCG@10 -silent
ELAPSED_TIME=$(((date +%s) - $START_TIME))
(date +%s) > .log/cur_time
python -c "print(int(open(\".log/cur_time\").read()) - int(open(\".log/start_time\").read()))"

(date +%s) > .log/start_time
java -jar ../RankLib-2.10.jar -train ../data/MSLR-WEB10K/Fold1/train.txt -ranker 7 -metric2t NDCG@10 -silent
ELAPSED_TIME=$(((date +%s) - $START_TIME))
(date +%s) > .log/cur_time
python -c "print(int(open(\".log/cur_time\").read()) - int(open(\".log/start_time\").read()))"

echo Fold2

(date +%s) > .log/start_time
java -jar ../RankLib-2.10.jar -train ../data/MSLR-WEB10K/Fold2/train.txt -ranker 0 -metric2t NDCG@10 -silent
ELAPSED_TIME=$(((date +%s) - $START_TIME))
(date +%s) > .log/cur_time
python -c "print(int(open(\".log/cur_time\").read()) - int(open(\".log/start_time\").read()))"

(date +%s) > .log/start_time
java -jar ../RankLib-2.10.jar -train ../data/MSLR-WEB10K/Fold2/train.txt -ranker 1 -metric2t NDCG@10 -silent
(date +%s) > .log/cur_time
python -c "print(int(open(\".log/cur_time\").read()) - int(open(\".log/start_time\").read()))"

(date +%s) > .log/start_time
java -jar ../RankLib-2.10.jar -train ../data/MSLR-WEB10K/Fold2/train.txt -ranker 2 -metric2t NDCG@10 -silent
ELAPSED_TIME=$(((date +%s) - $START_TIME))
(date +%s) > .log/cur_time
python -c "print(int(open(\".log/cur_time\").read()) - int(open(\".log/start_time\").read()))"

(date +%s) > .log/start_time
java -jar ../RankLib-2.10.jar -train ../data/MSLR-WEB10K/Fold2/train.txt -ranker 3 -metric2t NDCG@10 -silent
ELAPSED_TIME=$(((date +%s) - $START_TIME))
(date +%s) > .log/cur_time
python -c "print(int(open(\".log/cur_time\").read()) - int(open(\".log/start_time\").read()))"

(date +%s) > .log/start_time
java -jar ../RankLib-2.10.jar -train ../data/MSLR-WEB10K/Fold2/train.txt -ranker 3 -metric2t NDCG@10 -silent
ELAPSED_TIME=$(((date +%s) - $START_TIME))
(date +%s) > .log/cur_time
python -c "print(int(open(\".log/cur_time\").read()) - int(open(\".log/start_time\").read()))"

(date +%s) > .log/start_time
java -jar ../RankLib-2.10.jar -train ../data/MSLR-WEB10K/Fold2/train.txt -ranker 6 -metric2t NDCG@10 -silent
ELAPSED_TIME=$(((date +%s) - $START_TIME))
(date +%s) > .log/cur_time
python -c "print(int(open(\".log/cur_time\").read()) - int(open(\".log/start_time\").read()))"

(date +%s) > .log/start_time
java -jar ../RankLib-2.10.jar -train ../data/MSLR-WEB10K/Fold2/train.txt -ranker 7 -metric2t NDCG@10 -silent
ELAPSED_TIME=$(((date +%s) - $START_TIME))
(date +%s) > .log/cur_time
python -c "print(int(open(\".log/cur_time\").read()) - int(open(\".log/start_time\").read()))"

echo Fold3

(date +%s) > .log/start_time
java -jar ../RankLib-2.10.jar -train ../data/MSLR-WEB10K/Fold3/train.txt -ranker 0 -metric2t NDCG@10 -silent
ELAPSED_TIME=$(((date +%s) - $START_TIME))
(date +%s) > .log/cur_time
python -c "print(int(open(\".log/cur_time\").read()) - int(open(\".log/start_time\").read()))"

(date +%s) > .log/start_time
java -jar ../RankLib-2.10.jar -train ../data/MSLR-WEB10K/Fold3/train.txt -ranker 1 -metric2t NDCG@10 -silent
(date +%s) > .log/cur_time
python -c "print(int(open(\".log/cur_time\").read()) - int(open(\".log/start_time\").read()))"

(date +%s) > .log/start_time
java -jar ../RankLib-2.10.jar -train ../data/MSLR-WEB10K/Fold3/train.txt -ranker 2 -metric2t NDCG@10 -silent
ELAPSED_TIME=$(((date +%s) - $START_TIME))
(date +%s) > .log/cur_time
python -c "print(int(open(\".log/cur_time\").read()) - int(open(\".log/start_time\").read()))"

(date +%s) > .log/start_time
java -jar ../RankLib-2.10.jar -train ../data/MSLR-WEB10K/Fold3/train.txt -ranker 3 -metric2t NDCG@10 -silent
ELAPSED_TIME=$(((date +%s) - $START_TIME))
(date +%s) > .log/cur_time
python -c "print(int(open(\".log/cur_time\").read()) - int(open(\".log/start_time\").read()))"

(date +%s) > .log/start_time
java -jar ../RankLib-2.10.jar -train ../data/MSLR-WEB10K/Fold3/train.txt -ranker 3 -metric2t NDCG@10 -silent
ELAPSED_TIME=$(((date +%s) - $START_TIME))
(date +%s) > .log/cur_time
python -c "print(int(open(\".log/cur_time\").read()) - int(open(\".log/start_time\").read()))"

(date +%s) > .log/start_time
java -jar ../RankLib-2.10.jar -train ../data/MSLR-WEB10K/Fold3/train.txt -ranker 6 -metric2t NDCG@10 -silent
ELAPSED_TIME=$(((date +%s) - $START_TIME))
(date +%s) > .log/cur_time
python -c "print(int(open(\".log/cur_time\").read()) - int(open(\".log/start_time\").read()))"

(date +%s) > .log/start_time
java -jar ../RankLib-2.10.jar -train ../data/MSLR-WEB10K/Fold3/train.txt -ranker 7 -metric2t NDCG@10 -silent
ELAPSED_TIME=$(((date +%s) - $START_TIME))
(date +%s) > .log/cur_time
python -c "print(int(open(\".log/cur_time\").read()) - int(open(\".log/start_time\").read()))"

echo Fold4

(date +%s) > .log/start_time
java -jar ../RankLib-2.10.jar -train ../data/MSLR-WEB10K/Fold4/train.txt -ranker 0 -metric2t NDCG@10 -silent
ELAPSED_TIME=$(((date +%s) - $START_TIME))
(date +%s) > .log/cur_time
python -c "print(int(open(\".log/cur_time\").read()) - int(open(\".log/start_time\").read()))"

(date +%s) > .log/start_time
java -jar ../RankLib-2.10.jar -train ../data/MSLR-WEB10K/Fold4/train.txt -ranker 1 -metric2t NDCG@10 -silent
(date +%s) > .log/cur_time
python -c "print(int(open(\".log/cur_time\").read()) - int(open(\".log/start_time\").read()))"

(date +%s) > .log/start_time
java -jar ../RankLib-2.10.jar -train ../data/MSLR-WEB10K/Fold4/train.txt -ranker 2 -metric2t NDCG@10 -silent
ELAPSED_TIME=$(((date +%s) - $START_TIME))
(date +%s) > .log/cur_time
python -c "print(int(open(\".log/cur_time\").read()) - int(open(\".log/start_time\").read()))"

(date +%s) > .log/start_time
java -jar ../RankLib-2.10.jar -train ../data/MSLR-WEB10K/Fold4/train.txt -ranker 3 -metric2t NDCG@10 -silent
ELAPSED_TIME=$(((date +%s) - $START_TIME))
(date +%s) > .log/cur_time
python -c "print(int(open(\".log/cur_time\").read()) - int(open(\".log/start_time\").read()))"

(date +%s) > .log/start_time
java -jar ../RankLib-2.10.jar -train ../data/MSLR-WEB10K/Fold4/train.txt -ranker 3 -metric2t NDCG@10 -silent
ELAPSED_TIME=$(((date +%s) - $START_TIME))
(date +%s) > .log/cur_time
python -c "print(int(open(\".log/cur_time\").read()) - int(open(\".log/start_time\").read()))"

(date +%s) > .log/start_time
java -jar ../RankLib-2.10.jar -train ../data/MSLR-WEB10K/Fold4/train.txt -ranker 6 -metric2t NDCG@10 -silent
ELAPSED_TIME=$(((date +%s) - $START_TIME))
(date +%s) > .log/cur_time
python -c "print(int(open(\".log/cur_time\").read()) - int(open(\".log/start_time\").read()))"

(date +%s) > .log/start_time
java -jar ../RankLib-2.10.jar -train ../data/MSLR-WEB10K/Fold4/train.txt -ranker 7 -metric2t NDCG@10 -silent
ELAPSED_TIME=$(((date +%s) - $START_TIME))
(date +%s) > .log/cur_time
python -c "print(int(open(\".log/cur_time\").read()) - int(open(\".log/start_time\").read()))"

echo Fold5

(date +%s) > .log/start_time
java -jar ../RankLib-2.10.jar -train ../data/MSLR-WEB10K/Fold5/train.txt -ranker 0 -metric2t NDCG@10 -silent
ELAPSED_TIME=$(((date +%s) - $START_TIME))
(date +%s) > .log/cur_time
python -c "print(int(open(\".log/cur_time\").read()) - int(open(\".log/start_time\").read()))"

(date +%s) > .log/start_time
java -jar ../RankLib-2.10.jar -train ../data/MSLR-WEB10K/Fold5/train.txt -ranker 1 -metric2t NDCG@10 -silent
(date +%s) > .log/cur_time
python -c "print(int(open(\".log/cur_time\").read()) - int(open(\".log/start_time\").read()))"

(date +%s) > .log/start_time
java -jar ../RankLib-2.10.jar -train ../data/MSLR-WEB10K/Fold5/train.txt -ranker 2 -metric2t NDCG@10 -silent
ELAPSED_TIME=$(((date +%s) - $START_TIME))
(date +%s) > .log/cur_time
python -c "print(int(open(\".log/cur_time\").read()) - int(open(\".log/start_time\").read()))"

(date +%s) > .log/start_time
java -jar ../RankLib-2.10.jar -train ../data/MSLR-WEB10K/Fold5/train.txt -ranker 3 -metric2t NDCG@10 -silent
ELAPSED_TIME=$(((date +%s) - $START_TIME))
(date +%s) > .log/cur_time
python -c "print(int(open(\".log/cur_time\").read()) - int(open(\".log/start_time\").read()))"

(date +%s) > .log/start_time
java -jar ../RankLib-2.10.jar -train ../data/MSLR-WEB10K/Fold5/train.txt -ranker 3 -metric2t NDCG@10 -silent
ELAPSED_TIME=$(((date +%s) - $START_TIME))
(date +%s) > .log/cur_time
python -c "print(int(open(\".log/cur_time\").read()) - int(open(\".log/start_time\").read()))"

(date +%s) > .log/start_time
java -jar ../RankLib-2.10.jar -train ../data/MSLR-WEB10K/Fold5/train.txt -ranker 6 -metric2t NDCG@10 -silent
ELAPSED_TIME=$(((date +%s) - $START_TIME))
(date +%s) > .log/cur_time
python -c "print(int(open(\".log/cur_time\").read()) - int(open(\".log/start_time\").read()))"

(date +%s) > .log/start_time
java -jar ../RankLib-2.10.jar -train ../data/MSLR-WEB10K/Fold5/train.txt -ranker 7 -metric2t NDCG@10 -silent
ELAPSED_TIME=$(((date +%s) - $START_TIME))
(date +%s) > .log/cur_time
python -c "print(int(open(\".log/cur_time\").read()) - int(open(\".log/start_time\").read()))"
