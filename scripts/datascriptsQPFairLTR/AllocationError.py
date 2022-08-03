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
import pickle
from matplotlib.ticker import FuncFormatter
import json
import random
import utils.dataset as dataset
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
                        default="MQ2008",
                        help="Name of dataset to sample from.")
    parser.add_argument("--dataset_info_path", type=str,
                        default="LTRlocal_dataset_info.txt",
                        help="Path to dataset info file.")
    parser.add_argument("--fold_id", type=int,
                        help="Fold number to select, modulo operator is applied to stay in range.",
                        default=1)
    parser.add_argument("--query_least_size", type=int,
                        default=5,
                        help="query_least_size, filter out queries with number of docs less than this number.")
    parser.add_argument("--rankListLength", type=int,
                    help="Maximum number of items that can be displayed.",
                    default=5)
    parser.add_argument("--fairness_strategy", type=str,
                        choices=['FairCo', 'FairCo_multip.',"onlyFairness", 'QPFair','GradFair',"Randomk","FairK",\
                            "ExploreK","Topk","FairCo_maxnorm","QPFair","QPFair-Horiz.","ILP","LP"],
                        default="GradFair",
                        help="fairness_strategy, available choice is ['FairCo', 'FairCo_multip.', 'QPFair','GradFair','Randomk','Topk']")
    parser.add_argument("--fairness_tradeoff_param", type=float,
                            default=1.0,
                            help="fairness_tradeoff_param")
    parser.add_argument("--relvance_strategy", type=str,
                        choices=['TrueAverage',"NNmodel","EstimatedAverage"],
                        default="EstimatedAverage",
                        help="relvance_strategy, available choice is ['TrueAverage', 'NNmodel.', 'EstimatedAverage']")
    parser.add_argument("--exploration_strategy", type=str,
                        choices=['MarginalUncertainty',None],
                        default='MarginalUncertainty',
                        help="exploration_strategy, available choice is ['MarginalUncertainty', None]")
    parser.add_argument("--exploration_tradeoff_param", type=float,
                            default=0.0,
                            help="exploration_tradeoff_param")
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
                    help="how many future session we want consider in advance, only works if we use QPFair strategy.")
    parser.add_argument("--progressbar",  type=str2bool, nargs='?',
                    const=True, default=True,
                    help="use progressbar or not.")
    # args = parser.parse_args()
    args = parser.parse_args(args=[]) # for debug
    argsDict=vars(args)
    # print(args)
    # load the data and filter out queries with number of documents less than query_least_size.
    data = dataset.get_data(args.dataset_name,
                  args.dataset_info_path,
                  args.fold_id,
                  args.query_least_size,
                  relvance_strategy=args.relvance_strategy,\
                    voidFeature=True)
    positionBias=sim.getpositionBias(args.rankListLength,args.positionBiasSeverity)
    assert args.rankListLength>=args.query_least_size, print("For simplicity, the ranked list length should be greater than doc length")


    n_futureSessions=[1,5,10,15,20,30,50,100,200,500]
    MAError=[]
    MAError_max=[]
    ErrDict=defaultdict(list)
    AllocationFcn={"Horizontal":rnk.getHorizontalRanking,"Vertical":rnk.getVerticalRanking}
    queryRndSeed=np.random.default_rng(args.random_seed) 
    datasplit=data.test
    OutputPath=os.path.join(args.log_dir,"AllocationError")
    SavePath=os.path.join(OutputPath,args.dataset_name+"AllocError.pkl")
    if  os.path.isfile(SavePath): 
        with open(SavePath, 'rb') as f:
            ErrDict = pickle.load(f)
    else:
        for n_futureSession in n_futureSessions: 
            queriesList=datasplit.queriesList
            error=defaultdict(list)
            for qid in progressbar(queriesList):
                qExpVector=datasplit.query_values_from_vector(qid,datasplit.exposure)
                qRel=datasplit.query_values_from_vector(qid,datasplit.label_vector)
                QuotaEachItem=rnk.getQuotaEachItemNDCG(qExpVector,qRel,positionBias,n_futureSession,args.fairness_tradeoff_param)
                QuotaEachItemSum=QuotaEachItem.sum()
                QuotaEachItemOrig=np.zeros_like(QuotaEachItem)
                QuotaEachItemOrig[:]=QuotaEachItem
                # ExpoBackwardCum=rnk.getExpoBackwardCum(n_futureSession,args.rankListLength,positionBias)
                # ranking=rnk.getVerticalRanking(qRel,args.rankListLength,n_futureSession,QuotaEachItem,positionBias)
                for key,fcn in AllocationFcn.items():
                    ranking=fcn(qRel,args.rankListLength,n_futureSession,QuotaEachItem,positionBias)
                    ranking=np.array(ranking).astype(np.int)
                    qExpVectorResult=np.zeros_like(qExpVector)
                    np.add.at(qExpVectorResult,ranking,positionBias)
                    # print(QuotaEachItem.sum())
                    assert np.isclose(qExpVectorResult.sum(),QuotaEachItemSum)
                    error[key+"_MAE"].append(np.sum(np.abs(QuotaEachItemOrig-qExpVectorResult)/qExpVectorResult.sum()))
                    # MAError_max.append(np.max(np.abs(QuotaEachItemOrig,qExpVectorResult)))
                    # ErrDict[key+"_ErrMaxAbs"].append(np.max(np.abs(QuotaEachItemOrig-qExpVectorResult)))
                    # ErrDict[key+"_ErrMaxRelative"].append(np.max(np.abs(QuotaEachItemOrig-qExpVectorResult))/qExpVectorResult.mean())
            for key,fcn in AllocationFcn.items():
                ErrDict[key+"_MAE"].append(np.mean(error[key+"_MAE"]))
        with open(SavePath, 'wb') as f:
            pickle.dump(ErrDict, f)

    fig, ax = plt.subplots(figsize=(6.4,2.4))
    for key,value in ErrDict.items():
        if "_MAE" not in key:
            continue
        key=key.split("_")[0]
        plt.plot(n_futureSessions,value,label=key)
    # plt.plot(n_futureSessions,MAError_max1,label="Sto")
    # plt.plot(n_futureSessions,MAError_max2,)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
    ax.set_xlabel("The number of planning session.")
    ax.set_ylabel("Allocation Error")
    # ax.set_title("Error of exposure by vertical allocation")
    # axs[ind].set_xscale("log")
    # axs[0].set_yscale("symlog")
    # ax.legend(bbox_to_anchor=(1.1, 1.05))    
    ax.legend()    
    
    os.makedirs(OutputPath,exist_ok=True)
    fig.savefig(os.path.join(OutputPath,args.dataset_name+"AllocationError.pdf"), dpi=600, bbox_inches = 'tight', pad_inches = 0.05)
    plt.close(fig)