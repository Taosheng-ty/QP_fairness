import sys
import os
import pandas as pd
import matplotlib.pyplot as plt 
sys.path.append("/home/taoyang/research/Tao_lib/BEL/src/BatchExpLaunch")
import results_org as results_org
scriptPath=os.path.dirname(os.path.abspath(__file__))
os.chdir(scriptPath+"/../..")
path_root="localOutput/Feb192022DataTrueAver/"
# path_root="localOutput/Feb182022Data/"
# path_root="localOutput/Feb192022DataEstimatedAverage/"
path_root="localOutput/Feb222022Data/relvance_strategy_EstimatedAverage"
# path_root="localOutput/Feb222022Data/relvance_strategy_TrueAverage"
path_root="localOutput/Mar292022Data20Docs/relvance_strategy_TrueAverage"
path_root="localOutput/Mar292022Data20Docs/relvance_strategy_EstimatedAverage"
step=19  
data_rename={            
            "Movie":"Movie",\
            # "News":"News",\
            # "MSLR-WEB30k":"MSLR-WEB30k",\
            # "MSLR-WEB10k":"MSLR-WEB10k",\
            # "Webscope_C14_Set1":"Webscope_C14_Set1",\
            # "istella-s":"istella-s"
}
metric_name=[["disparity",'NDCG_1_aver'],["disparity",'NDCG_3_aver'],["disparity",'NDCG_5_aver']]
metric_name_dict={"NDCG_1_aver":"NDCG@1","NDCG_3_aver":"NDCG@3","NDCG_5_aver":"NDCG@5","NDCG_10_aver":"NDCG@10",}
result_list=[]

# result_path=os.path.join(path_root,"result")
# for datasets,data_name_cur in data_rename.items():
#     fig, axs = plt.subplots(len(metric_name),figsize=(5,3*len(metric_name)), sharex=True)
#     result_validated={}
#     datasets="dataset_"+datasets
#     path=os.path.join(path_root,datasets)


positionBiasSeverities=[
    # "positionBiasSeverity_0",
    "positionBiasSeverity_1",
    # "positionBiasSeverity_2"
    ]
result_list=[]
for positionBiasSeverity in positionBiasSeverities:
    OutputPath=os.path.join(path_root,"result")
    for datasets,data_name_cur in data_rename.items():
        fig, axs = plt.subplots(len(metric_name),figsize=(5,3*len(metric_name)), sharex=True)
        result_validated={}
        datasets="dataset_name_"+datasets
        resultPath=os.path.join(path_root,positionBiasSeverity,datasets)

        if not os.path.isdir(resultPath):
            # print(path)       
            continue
        # result,result_mean=results_org.get_result_df(resultPath,groupby="iterations",rerun=True)
        result,result_mean=results_org.get_result_df(resultPath,groupby="iterations")
        # result_validated["FairCo"]=result["fairness_strategy_FairCo"]
        # result_validated["FairCo_maxnorm"]=result["fairness_strategy_FairCo_maxnorm"]
        # result_validated["FairCo_multip."]=result["fairness_strategy_FairCo_multip."]
        result_validated["Eff.+Fair.+Uncert."]= results_org.getGrandchildNode(result["fairness_strategy_FairCo_average"],"exploration_tradeoff_param_20")

        result_validated["Eff.+Fair."]= results_org.getGrandchildNode(result["fairness_strategy_FairCo_average"],"exploration_tradeoff_param_0.0")
        result_validated["Eff.+Uncert."]= result["fairness_strategy_FairCo_average"]["fairness_tradeoff_param_0.0"]
        result_validated["Fair.+Uncert."]= result["fairness_strategy_FairCo_average"]["fairness_tradeoff_param_1"]

        # result_validated["Fair.+Uncert."]= result["fairness_strategy_FairCo_average"]["fairness_tradeoff_param_0.0"]
        # result_validated["LP"]=result["fairness_strategy_LP"]
        # result_validated["ILP"]=result["fairness_strategy_ILP"]
        # result_validated["FairCo_averageexp10"]=results_org.getGrandchildNode(result["fairness_strategy_FairCo_average"],"exploration_tradeoff_param_10")
        # result_validated["QPfair_5"]=result["fairness_strategy_QPfair"]["n_futureSession_5"]
        # result_validated["QPfairQuota"]=result["fairness_strategy_QPfair"]["n_futureSession_200"]
        # result_validated["QPfair_100"]=result["fairness_strategy_QPfair"]["n_futureSession_100"]
        # result_validated["QPfairNDCG_20"]=result["fairness_strategy_QPfairNDCG"]["n_futureSession_20"]
        # result_validated["QPfairNDCG_100"]=result["fairness_strategy_QPfairNDCG"]["n_futureSession_100"]
        # result_validated["QPfairNDCG"]=result["fairness_strategy_QPfairNDCG"]["n_futureSession_200"]
        # result_validated["HQPfair"]=result["fairness_strategy_Hybrid"]["n_futureSession_20"]
        # result_validated["HQPfair5"]=result["fairness_strategy_Hybrid"]["n_futureSession_5"]
        # result_validated["HQPfair_100"]=result["fairness_strategy_Hybrid"]["n_futureSession_100"]
        # result_validated["HQPfair_200"]=result["fairness_strategy_Hybrid"]["n_futureSession_200"]
        # result_validated["QPfair_100"]=result["fairness_strategy_QPfair"]["n_futureSession_100"]
        result_validatedScatter={}
        result_validatedScatter["only Eff."]=result["fairness_strategy_Topk"]
        # result_validatedScatter["RandomK"]=result["fairness_strategy_Randomk"]
        result_validatedScatter["only Fair."]=result["fairness_strategy_FairK"]
        result_validatedScatter["only Uncert."]=result["fairness_strategy_ExploreK"]
        for ind,metrics in enumerate(metric_name):
            results_org.TradeoffPlot(result_validated, metrics,ax=axs[ind],step=step)
            results_org.TradeoffScatter(result_validatedScatter, metrics,ax=axs[ind],step=step)
            axs[ind].set_ylabel(metric_name_dict[metrics[1]])
            axs[ind].set_xlabel(metrics[0])
            axs[ind].set_title(data_name_cur)
            # axs[ind].set_xscale("log")
            # axs[0].set_yscale("symlog")
            axs[ind].legend(bbox_to_anchor=(1.1, 1.05))    
        fig.savefig(os.path.join(OutputPath,positionBiasSeverity+data_name_cur+"Abalation_tradeoffplot.pdf"), dpi=600, bbox_inches = 'tight', pad_inches = 0.05)
        plt.close(fig)