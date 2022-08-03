import sys
import os
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.ticker as mticker
sys.path.append("/home/taoyang/research/Tao_lib/BEL/src/BatchExpLaunch")
import results_org as results_org
scriptPath=os.path.dirname(os.path.abspath(__file__))
os.chdir(scriptPath+"/../..")
path_root="localOutput/Feb192022DataTrueAver/"
path_root="localOutput/Feb192022DataEstimatedAverage/"
# path_root="localOutput/Feb222022Data/relvance_strategy_EstimatedAverage"
# path_root="localOutput/Feb222022Data/relvance_strategy_TrueAverage"
path_root="localOutput/Mar292022Data20Docs/relvance_strategy_EstimatedAverage"
path_root="localOutput/Mar292022Data20Docs/relvance_strategy_TrueAverage"
path_root="localOutput/QPFairLTR/relvance_strategy_TrueAverage"
path_root="localOutput/QPFairLTRistella/relvance_strategy_TrueAverage"
path_root="localOutput/Apr30QPFairLTR/relvance_strategy_TrueAverage"
path_root="localOutput/July3QPFairLTR/relvance_strategy_TrueAverage"
step=19  
data_rename={            
            # "Movie":"Movie",\
            # "News":"News",\
            # "MSLR-WEB30k":"MSLR-WEB30k",\
            # "MSLR-WEB10k":"MSLR10k",\
            # "Webscope_C14_Set1":"Webscope_C14_Set1",\
            # "istella-s":"istella-s"，
            "MQ2008":"MQ2008",
            # "MQ2007":"MQ2007",
            # "istella-s":"ist",
}
metric_name=['test_NDCG_1_aver','test_NDCG_3_aver','test_NDCG_5_aver','test_NDCG_1_cumu','test_NDCG_3_cumu','test_NDCG_5_cumu',"test_disparity"]
metric_name_dict={"test_NDCG_1_aver":"NDCG@1","test_NDCG_3_aver":"NDCG@3","test_NDCG_5_aver":"NDCG@5",\
    "test_NDCG_1_cumu":"cNDCG@1","test_NDCG_3_cumu":"cNDCG@3","test_NDCG_5_cumu":"cNDCG@5","test_disparity":"Unfairness"}
result_list=[]

result_path=os.path.join(path_root,"result")
for datasets,data_name_cur in data_rename.items():
    fig, axs = plt.subplots(len(metric_name),figsize=(5,3*len(metric_name)), sharex=True)
    result_validated={}
    datasets="dataset_"+datasets
    path=os.path.join(path_root,datasets)

positionBiasSeverities=[
    # "positionBiasSeverity_0",
    "positionBiasSeverity_1",
    # "positionBiasSeverity_2"
    ]
result_list=[]
for positionBiasSeverity in positionBiasSeverities:
    OutputPath=os.path.join(path_root,"result")
    for datasets,data_name_cur in data_rename.items():
        
        datasets="dataset_name_"+datasets
        resultPath=os.path.join(path_root,positionBiasSeverity,datasets)
        result_validated={}
        if not os.path.isdir(resultPath):
            # print(path)       
            continue
        result,result_mean=results_org.get_result_df(resultPath,groupby="iterations")
        # result_validated["QP"]=results_org.getGrandchildNode(result["fairness_strategy_QPfairNDCG"],"fairness_tradeoff_param_1.0")
        result_validated["QPFair"]=results_org.getGrandchildNode(result["fairness_strategy_QPFair"],"fairness_tradeoff_param_1.0")
        result_validated["QPFair-Horiz"]=results_org.getGrandchildNode(result["fairness_strategy_QPFair-Horiz."],"fairness_tradeoff_param_1.0")
        result_validated["QPFair"]=results_org.getGrandchildNode(result_validated["QPFair"],"exploration_tradeoff_param_0.0")
        result_validated["QPFair-Horiz"]=results_org.getGrandchildNode(result_validated["QPFair-Horiz"],"exploration_tradeoff_param_0.0")         
        # result_validated["QPfairQuota"]=results_org.getGrandchildNode(result["fairness_strategy_QPfair"],"fairness_tradeoff_param_1.0")

        for ind,metrics in enumerate(metric_name):
            fig, axs = plt.subplots()
            results_org.paramIterationPlot(result_validated, metrics,ax=axs,step=step)
            axs.set_ylabel(metric_name_dict[metrics])
            axs.set_xlabel("The number of future sessions to consider.")
            # axs.set_title(data_name_cur)
            axs.set_xscale("log")
            axs.xaxis.set_major_formatter(mticker.ScalarFormatter())
            axs.xaxis.get_major_formatter().set_scientific(False)
            axs.xaxis.get_major_formatter().set_useOffset(False)
            # axs[0].set_yscale("symlog")
            # axs.legend(bbox_to_anchor=(1.1, 1.05))    
            axs.legend()
            fig.savefig(os.path.join(OutputPath,metrics+positionBiasSeverity+data_name_cur+"performance_along_futureQP.pdf"), dpi=600, bbox_inches = 'tight', pad_inches = 0.05)
            plt.close(fig)