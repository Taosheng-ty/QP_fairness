import numpy as np
import utils.dataset as dataset
import utils.simulation as sim
import utils.ranking as rnk
import utils.evaluation as evl
from collections import defaultdict
from progressbar import progressbar
import argparse
from str2bool import str2bool
import json
import os
import random
import sys
sys.path.append("/home/taoyang/research/Tao_lib/BEL/src/BatchExpLaunch")
import results_org as results_org
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str,
                        default="localOutput/",
                        help="Path to result logs")
    parser.add_argument("--dataset", type=str,
                        default="MQ2007",
                        help="Name of dataset to sample from.")
    parser.add_argument("--dataset_info_path", type=str,
                        default="local_dataset_info.txt",
                        help="Path to dataset info file.")
    parser.add_argument("--fold_id", type=int,
                        help="Fold number to select, modulo operator is applied to stay in range.",
                        default=1)
    parser.add_argument("--query_least_size", type=int,
                        default=10,
                        help="query_least_size, filter out queries with number of docs less than this number.")
    parser.add_argument("--rankListLength", type=int,
                    help="Maximum number of items that can be displayed.",
                    default=10)
    parser.add_argument("--fairness_strategy", type=str,
                        choices=['FairCo', 'FairCo_multip.', 'QPfair','FairCo_average',"Randomk","Topk"],
                        default="QPfair",
                        help="fairness_strategy, available choice is ['FairCo', 'FairCo_multip.', 'QPfair','FairCo_average','Randomk','Topk']")
    parser.add_argument("--fairness_tradeoff_param", type=float,
                            default=1.0,
                            help="fairness_tradeoff_param")
    parser.add_argument("--random_seed", type=int,
                    default=0,
                    help="random seed for reproduction")
    parser.add_argument("--positionBiasSeverity", type=int,
                    help="Severity of positional bias",
                    default=1)
    parser.add_argument("--n_iteration", type=int,
                    default=10**4,
                    help="how many iteractions to simulate")
    parser.add_argument("--n_futureSession", type=int,
                    default=20,
                    help="how many future session we want consider in advance, only works if we use QPfair strategy.")
    parser.add_argument("--progressbar",  type=str2bool, nargs='?',
                    const=True, default=True,
                    help="use progressbar or not.")
    args = parser.parse_args()
    # args = parser.parse_args(args=[]) # for debug
    # load the data and filter out queries with number of documents less than query_least_size.
    data = dataset.get_data(args.dataset,
                  args.dataset_info_path,
                  args.fold_id,
                  args.query_least_size)
    # begin simulation
    Logging=results_org.getLogging()
    positionBias=sim.getpositionBias(args.rankListLength,args.positionBiasSeverity) 
    NDCGcutoffs=[i for i in [1,3,5,10,20] if i<=args.rankListLength]
    assert args.rankListLength>=args.query_least_size, print("For simplicity, the ranked list length should be greater than doc length")
    queryRndSeed=np.random.default_rng(args.random_seed) 
    random.seed(args.random_seed)
    OutputDict=defaultdict(list)
    NDCGDict=defaultdict(list)
    evalIterations=np.linspace(0, args.n_iteration-1, num=21,endpoint=True).astype(np.int32)[1:]
    iterationsGenerator=progressbar(range(args.n_iteration)) if args.progressbar else range(args.n_iteration)
    for iteration in iterationsGenerator:
        # sample data split and a query
        qid,dataSplit=sim.sample_queryFromdata(data,queryRndSeed)
        if iteration in evalIterations:
            Logging.info("current iteration"+str(iteration))
            OutputDict["iterations"].append(iteration)
            evl.outputNDCG(NDCGDict,OutputDict)
            evl.outputFairness(data,OutputDict)
        if dataSplit.name !="test":
            continue
        # get a ranking according to fairness strategy
        ranking=rnk.get_ranking(qid,\
                                dataSplit,\
                                args.fairness_strategy,\
                                args.fairness_tradeoff_param,\
                                args.rankListLength,
                                args.n_futureSession,
                                positionBias)
        # update exposure statistics according to ranking
        rnk.updateExposure(qid,dataSplit,ranking,positionBias)
        # calculate metrics ndcg and unfairness.
        evl.Update_NDCG_multipleCutoffs(ranking,qid,dataSplit,positionBias,NDCGcutoffs,NDCGDict)

    #write the results.
    os.makedirs(args.log_dir,exist_ok=True)
    logPath=args.log_dir+"/result.jjson"
    print('Writing results to %s' % OutputDict)
    with open(logPath, 'w') as f:
        json.dump(OutputDict, f)
        