import sys
import os.path
import optparse


parameter_adarank = {
    "Name" : "AdaRank",
    "rankerNumber" : 3,
    "-round" : [500, 1000, 2000],
    "-tolerance" : [0.01, 0.001, 0.002, 0.005],
    "-max" : [2, 5, 10]
}

parameter_lambda = {
    "Name" : "LambdaMart",
    "rankerNumber" : 6,
    "-tree" : [1000, 2000],
    "-leaf" : [5, 10, 15],
    "-shrinkage" : [0.1, 0.01]
}

parameter_ranknet = {
    "Name" : "RankNet",
    "rankerNumber" : 1,
    "-epoch" : [50, 100, 200, 300],
    "-layer" : [1, 5, 10],
    "-node" : [10, 20, 100]
}

parameter_listnet = {
    "Name" : "ListNet",
    "rankerNumber" : 7,
    "-epoch" : [1000, 1500, 2000],
    "-lr" : [0.0001, "0.00001"]
}

dics = [parameter_lambda, parameter_adarank, parameter_ranknet, parameter_listnet]

parameter_adarank_test = {
    "Name" : "AdaRank",
    "rankerNumber" : 3,
    "-round" : [1],
}

parameter_lambda_test = {
    "Name" : "LambdaMart",
    "rankerNumber" : 6,
    "-tree" : [10],
}

parameter_ranknet_test = {
    "Name" : "RankNet",
    "rankerNumber" : 1,
    "-epoch" : [1],
}

parameter_listnet_test = {
    "Name" : "ListNet",
    "rankerNumber" : 7,
    "-epoch" : [1],
}

dics_test = [parameter_lambda_test, parameter_adarank_test, parameter_ranknet_test, parameter_listnet_test]

sbatch_pre = """#!/bin/bash
#SBATCH --ntasks 4
#SBATCH --qos bbdefault
#SBATCH --mail-type ALL
#SBATCH --mem 45G # 45GB
#SBATCH --error=slurm-%j.err
#SBATCH --time=72:00:00
set -e
module purge; module load bluebear # this line is required
module load Java/1.8.0_92
echo "Java now ..."
"""


def options():
    parser = optparse.OptionParser()
    parser.add_option("-d", "--data", dest="data_dir",
      help="where to find the data DIRECTORY")
    parser.add_option("-l", "--datalabel", dest="data_label",
      help="tag the experiments by label")
    parser.add_option("-j", "--jarpath", dest="jar_path",
      help="where to look for the jar file")
    parser.add_option("-r", "--resultsdir", dest="results_dir",
      help="directory to store results")
    parser.add_option("-c", "--cluster", dest="cluster",
      help="run on the cluster")
    parser.add_option("-t", "--test", dest="test",
      help="run in test mode")

    (options, args) = parser.parse_args()
    return (options, args)


def main(data_dir,data_label,jar_path,results_dir,cluster,test):
    simpleJobCounter = 0
    if not os.path.exists('run_scripts'):
        os.makedirs('run_scripts')
    if test == "1": run_dics = dics_test
    if test != "1": run_dics = dics
    for dic in run_dics:
        for metric in ["NDCG@10" , "MAP"]:
            for fold in ["Fold1", "Fold2", "Fold3", "Fold4", "Fold5"]:
                for key in dic.keys():
                    if key not in ["Name","rankerNumber"]:
                        for gridValue in dic[key]:
                            simpleJobCounter += 1
                            scriptfile = 'run_scripts{}{}_{}_{}_{}_{}_train.sh'.format(os.path.sep,data_label,dic["Name"],metric,fold,str(simpleJobCounter).zfill(6))
                            with open(scriptfile, "w") as bashfile:
                                if cluster == "1":
                                    bashfile.write(sbatch_pre)

                                outfile = 'echo "{} {} {} {}";\n'.format(data_label,dic["Name"],metric,fold,key)
                                outfile += 'java -jar {} -silent -kcv 5'.format(jar_path)
                                outfile += " -train {}{}{}{}train.txt".format(data_dir,os.path.sep,fold,os.path.sep)
                                outfile += " -test {}{}{}{}test.txt".format(data_dir,os.path.sep,fold,os.path.sep)
                                outfile += ' -ranker {}'.format(dic["rankerNumber"])
                                outfile += ' -metric2t {}'.format(metric)
                                outfile += " {} {}".format(key,gridValue)
                                outfile += " > {}{}{}{}{}{}.out".format(results_dir,metric,fold,key,gridValue,dic["Name"])
                                outfile += " ;"
                                # outfile += " &"
                                bashfile.write(outfile)
                                bashfile.write("\n")
                            print('sbatch {}'.format(scriptfile))

if __name__ == '__main__':
    (options, args) = options()
    data_dir    = options.data_dir
    data_label  = options.data_label
    jar_path    = '{}{}RankLib-2.10.jar'.format(options.jar_path,os.path.sep)
    results_dir = '{}{}{}{}'.format(options.results_dir,os.path.sep,data_label,os.path.sep)
    # data_dir
    if not os.path.exists(data_dir):
        print('Data path <<{}>> does not exist'.format(data_dir))
        sys.exit(0)
    if not os.path.exists(jar_path):
        print('JAR path <<{}>> not found'.format(jar_path))
        sys.exit(0)
    if not os.path.exists(results_dir):
        try:
            os.makedirs(results_dir)
        except Exception as e:
            print('Could not create output directory <<{}>>'.format(results_dir))
            sys.exit(0)
    main(data_dir,data_label,jar_path,results_dir,options.cluster,options.test)
