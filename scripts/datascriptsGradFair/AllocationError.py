import os
import sys
scriptPath=os.path.dirname(os.path.abspath(__file__))
os.chdir(scriptPath+"/../..")
sys.path.append(scriptPath+"/../..")
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
import random

import time
import matplotlib.pyplot as plt
sys.path.append("/home/taoyang/research/Tao_lib/BEL/src/BatchExpLaunch")
import results_org as results_org

def MAErr(a,b):
    return np.sum(np.abs(a-b))
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
                        choices=['FairCo', 'FairCo_multip.',"onlyFairness", 'QPfair','FairCo_average',"Randomk","FairK","ExploreK","Topk","Hybrid","FairCo_maxnorm","QPfairNDCG","ILP","LP"],
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
                            default=1.0,
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
                    default=200,
                    help="the Maximum number of docs")
    parser.add_argument("--progressbar",  type=str2bool, nargs='?',
                    const=True, default=True,
                    help="use progressbar or not.")
    # args = parser.parse_args()
    args = parser.parse_args(args=[]) # for debug
    argsDict=vars(args)
    # print(args)
    # load the data and filter out queries with number of documents less than query_least_size.
    data = data_utils.load_data(args.dataset_name,
                  args.dataset_info_path,
                  RandomSeed=args.random_seed,
                  relvance_strategy=args.relvance_strategy,
                  NumDocMaximum=args.NumDocMaximum
                  )
    args.rankListLength=data.getNumDoc() if args.rankListLength is None else args.rankListLength
    positionBias=sim.getpositionBias(args.rankListLength,args.positionBiasSeverity)
    qExpVector=data.exposure 
    qRel=data.getEstimatedAverageRelevance(None)
n_futureSessions=[1,5,10,15,20,30,50,100,200,500]
MAError=[]
MAError_max=[]
ErrDict=defaultdict(list)
AllocationFcn={"Horizontal":rnk.getHorizontalRanking,"Vertical":rnk.getVerticalRanking}
for n_futureSession in n_futureSessions:
    QuotaEachItem=rnk.getQuotaEachItemNDCG(qExpVector,qRel,positionBias,n_futureSession,args.fairness_tradeoff_param)
    QuotaEachItemSum=QuotaEachItem.sum()
    QuotaEachItemOrig=np.zeros_like(QuotaEachItem)
    QuotaEachItemOrig[:]=QuotaEachItem
    # ExpoBackwardCum=rnk.getExpoBackwardCum(n_futureSession,args.rankListLength,positionBias)
    # ranking=rnk.getVerticalRanking(qRel,args.rankListLength,n_futureSession,QuotaEachItem,positionBias)
    for key,fcn in AllocationFcn.items():
        ranking=rnk.getHorizontalRanking(qRel,args.rankListLength,n_futureSession,QuotaEachItem,positionBias)
        ranking=np.array(ranking).astype(np.int)
        qExpVectorResult=np.zeros_like(qExpVector)
        np.add.at(qExpVectorResult,ranking,positionBias)
        # print(QuotaEachItem.sum())
        assert np.isclose(qExpVectorResult.sum(),QuotaEachItemSum)
        ErrDict[key+"_MAE"].append(np.mean(np.abs(QuotaEachItemOrig-qExpVectorResult)/qExpVectorResult.mean())/2)
        # MAError_max.append(np.max(np.abs(QuotaEachItemOrig,qExpVectorResult)))
        ErrDict[key+"_ErrMaxAbs"].append(np.max(np.abs(QuotaEachItemOrig-qExpVectorResult)))
        ErrDict[key+"_ErrMaxRelative"].append(np.max(np.abs(QuotaEachItemOrig-qExpVectorResult))/qExpVectorResult.mean())
fig, ax = plt.subplots()
for key,value in ErrDict.items():
    if "_MAE" not in key:
        continue
    plt.plot(n_futureSessions,value,label=key)
# plt.plot(n_futureSessions,MAError_max1,label="Sto")
# plt.plot(n_futureSessions,MAError_max2,)
ax.set_xlabel("the number of future session. $nf$")
ax.set_ylabel("MAE")
ax.set_title("Error of exposure by vertical allocation")
# axs[ind].set_xscale("log")
# axs[0].set_yscale("symlog")
ax.legend(bbox_to_anchor=(1.1, 1.05))      
OutputPath=os.path.join(args.log_dir,"AllocationError")
os.makedirs(OutputPath,exist_ok=True)
fig.savefig(os.path.join(OutputPath,"AllocationError.pdf"), dpi=600, bbox_inches = 'tight', pad_inches = 0.05)
plt.close(fig)