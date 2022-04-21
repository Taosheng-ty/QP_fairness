import numpy as np
import utils.data_utils as data_utils
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
import time
sys.path.append("/home/taoyang/research/Tao_lib/BEL/src/BatchExpLaunch")
import results_org as results_org
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str,
                        default="localOutput/",
                        help="Path to result logs")
    parser.add_argument("--dataset_name", type=str,
                        default="Movie",
                        help="Name of dataset to sample from.")
    parser.add_argument("--dataset_info_path", type=str,
                        default="local_dataset_info.txt",
                        help="Path to dataset info file.")
    parser.add_argument("--rankListLength", type=int,
                    help="Maximum number of items that can be displayed.",
                    default=5)
    parser.add_argument("--fairness_strategy", type=str,
                        choices=['FairCo', 'FairCo_multip.',"onlyFairness", 'QPfair','FairCo_average',"Randomk","FairK",\
                            "ExploreK","Topk","Hybrid","FairCo_maxnorm","QPfairNDCG","QPfairNDCGHorizontal","ILP","LP"],
                        default='LP',
                        help="fairness_strategy, available choice is ['FairCo', 'FairCo_multip.', 'QPfair','FairCo_average','Randomk','Topk','Hybrid']")
    parser.add_argument("--relvance_strategy", type=str,
                        choices=['TrueAverage',"NNmodel","EstimatedAverage"],
                        default="TrueAverage",
                        help="relvance_strategy, available choice is ['TrueAverage', 'NNmodel.', 'EstimatedAverage']")
    parser.add_argument("--exploration_strategy", type=str,
                        choices=['MarginalUncertainty',None],
                        default='MarginalUncertainty',
                        help="exploration_strategy, available choice is ['MarginalUncertainty', None]")
    parser.add_argument("--exploration_tradeoff_param", type=float,
                            default=0.0,
                            help="exploration_tradeoff_param")
    parser.add_argument("--fairness_tradeoff_param", type=float,
                            default=0.5,
                            help="fairness_tradeoff_param")
    parser.add_argument("--random_seed", type=int,
                    default=0,
                    help="random seed for reproduction")
    parser.add_argument("--positionBiasSeverity", type=int,
                    help="Severity of positional bias",
                    default=1)
    parser.add_argument("--n_iteration", type=int,
                    default=3*10**3,
                    help="how many iteractions to simulate")
    parser.add_argument("--n_futureSession", type=int,
                    default=1,
                    help="how many future session we want consider in advance, only works if we use QPfair strategy.")
    parser.add_argument("--NumDocMaximum", type=int,
                    default=20,
                    help="the Maximum number of docs")
    parser.add_argument("--progressbar",  type=str2bool, nargs='?',
                    const=True, default=True,
                    help="use progressbar or not.")
    args = parser.parse_args()
    # args = parser.parse_args(args=[]) # for debug
    argsDict=vars(args)
    print(args)
    # load the data and filter out queries with number of documents less than query_least_size.
    data = data_utils.load_data(args.dataset_name,
                  args.dataset_info_path,
                  RandomSeed=args.random_seed,
                  relvance_strategy=args.relvance_strategy,
                  NumDocMaximum=args.NumDocMaximum
                  )
    # begin simulation
    Logging=results_org.getLogging()
    args.rankListLength=data.getNumDoc() if args.rankListLength is None else args.rankListLength
    positionBias=sim.getpositionBias(args.rankListLength,args.positionBiasSeverity) 
    NDCGcutoffs=[i for i in [1,3,5,10,20,30,50,100] if i<=args.rankListLength]
    random.seed(args.random_seed)
    clickRNG=np.random.default_rng(args.random_seed) 
    OutputDict=defaultdict(list)
    NDCGDict=defaultdict(list)
    evalIterations=np.linspace(0, args.n_iteration-1, num=21,endpoint=True).astype(np.int32)[1:]
    iterationsGenerator=progressbar(range(args.n_iteration)) if args.progressbar else range(args.n_iteration)
    start_time = time.time()
    for iteration in iterationsGenerator:
        # sample data split and a query
        if iteration in evalIterations:
            Logging.info("current iteration"+str(iteration))
            OutputDict["iterations"].append(iteration)
            OutputDict["time"].append(time.time()-start_time)
            evl.outputNDCG(NDCGDict,OutputDict)
            evl.outputFairnessData(data,OutputDict)
        # get a ranking according to fairness strategy.
        TrueRel,userFeature=next(data.DataGenerator)
        ranking=rnk.get_rankingFromData(data=data,\
                                        userFeature=userFeature,\
                                        positionBias=positionBias,\
                                        **argsDict)
        clicks=sim.generateClick(ranking,TrueRel,positionBias,clickRNG)
        # update exposure statistics according to ranking.
        data.updateStatistics(clicks,ranking,positionBias,userFeature)
        # calculate metrics ndcg and unfairness.
        evl.Update_NDCG_multipleCutoffsData(ranking,TrueRel,positionBias,NDCGcutoffs,NDCGDict)
    #write the results.
    os.makedirs(args.log_dir,exist_ok=True)
    logPath=args.log_dir+"/result.jjson"
    print(logPath)
    print('Writing results to %s' % OutputDict)
    with open(logPath, 'w') as f:
        json.dump(OutputDict, f)