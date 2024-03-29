import sys
import os
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib
font = {'family' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)
import config
from matplotlib import scale
scale.register_scale(config.Mylog2f)

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
path_root="localOutput/Apr262022LTR/relvance_strategy_EstimatedAverage"
path_root="localOutput/QPFairLTRistella/relvance_strategy_TrueAverage"
path_root="localOutput/Apr30QPFairLTR/relvance_strategy_TrueAverage"
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
metric_name=[["test_disparity",'test_NDCG_1_aver'],["test_disparity",'test_NDCG_3_aver'],\
    ["test_disparity",'test_NDCG_5_aver'],["test_disparity",'test_NDCG_1_cumu'],["test_disparity",'test_NDCG_3_cumu'],\
    ["test_disparity",'test_NDCG_5_cumu'],]
# metric_name=[["disparity",'NDCG_3_aver'],["disparity",'NDCG_5_aver']]

metric_name_dict={"test_NDCG_1_aver":"NDCG@1","test_NDCG_3_aver":"NDCG@3","test_NDCG_5_aver":"NDCG@5",\
    "test_NDCG_1_cumu":"cNDCG@1","test_NDCG_3_cumu":"cNDCG@3","test_NDCG_5_cumu":"cNDCG@5",}
result_list=[]
positionBiasSeverities=[
    # "positionBiasSeverity_0",
    "positionBiasSeverity_1",
    # "positionBiasSeverity_2"
    ]
for positionBiasSeverity in positionBiasSeverities:
    OutputPath=os.path.join(path_root,"result")
    for datasets,data_name_cur in data_rename.items():
        
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
        # result_validated["LP_1"]=result["fairness_strategy_LP"]["n_futureSession_1"]
        # if "fairness_strategy_LP" in result:
        #     result_validated["LP"]=result["fairness_strategy_LP"]["n_futureSession_100"]

        #     # result_validated["GradFair(Ours)"]=result["fairness_strategy_FairCo_average"]
        #     result_validated["ILP"]=result["fairness_strategy_ILP"]
        # for method in result_validated:
        #     result_validated[method]=results_org.getGrandchildNode(result_validated[method],"exploration_tradeoff_param_0.0")
        # result_validated["QPfairNDCG_500"]=result["fairness_strategy_QPfairNDCG"]["n_futureSession_500"]
        # result_validated["QPfairNDCG_500Hori"]=result["fairness_strategy_QPfairNDCGHorizontal"]["n_futureSession_500"]
        # result_validated["GradFair(Ours)_0"]=results_org.getGrandchildNode(result["fairness_strategy_GradFair"],"exploration_tradeoff_param_0.0")        
        # result_validated["GradFair(Ours)_0.1"]=results_org.getGrandchildNode(result["fairness_strategy_GradFair"],"exploration_tradeoff_param_0.1")
        # result_validated["GradFair(Ours)_0.5"]=results_org.getGrandchildNode(result["fairness_strategy_GradFair"],"exploration_tradeoff_param_0.5")
        # result_validated["GradFair(Ours)"]=results_org.getGrandchildNode(result["fairness_strategy_GradFair"],"exploration_tradeoff_param_10")
        # result_validated=results_org.reorderDict(result_validated,config.desiredGradFair)
        result_validated["GradFair(\\beta=0)"]=results_org.getGrandchildNode(result["fairness_strategy_GradFair"],"exploration_tradeoff_param_0.0")        
        # result_validated["GradFair(Ours)_0.5"]=results_org.getGrandchildNode(result["fairness_strategy_GradFair"],"exploration_tradeoff_param_0.5")        
        result_validated["GradFair(\\beta=1)"]=results_org.getGrandchildNode(result["fairness_strategy_GradFair"],"exploration_tradeoff_param_1")
        # result_validated["GradFair(Ours)_5"]=results_org.getGrandchildNode(result["fairness_strategy_GradFair"],"exploration_tradeoff_param_5")
        result_validated["GradFair(\\beta=10)"]=results_org.getGrandchildNode(result["fairness_strategy_GradFair"],"exploration_tradeoff_param_20")

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
        # result_validatedScatter={}
        # result_validatedScatter["TopK"]=result["fairness_strategy_Topk"]
        # result_validatedScatter["RandomK"]=result["fairness_strategy_Randomk"]
        # result_validatedScatter["FairK(Ours)"]=result["fairness_strategy_FairK"]
        # result_validatedScatter["ExploreK"]=result["fairness_strategy_ExploreK"]
        for ind,metrics in enumerate(metric_name):
            fig, axs = plt.subplots()
            results_org.TradeoffPlot(result_validated, metrics,ax=axs,step=step)
            # results_org.TradeoffScatter(result_validatedScatter, metrics,ax=axs,step=step)
            axs.set_ylabel(metric_name_dict[metrics[1]])
            axs.set_xlabel(metrics[0])
            # axs.set_title(data_name_cur)
            axs.set_xscale("log")
            axs.set_yscale("mylog2f")
            # axs[ind].set_xscale("log")
            # axs[0].set_yscale("symlog")
            # axs.legend(bbox_to_anchor=(1.1, 1.05)) 
            # axs.legend()   
            results_org.reorderLegend(config.desiredGradFair,axs)
            plt.setp(plt.gca().get_legend().get_texts(), fontsize='12')
            fig.savefig(os.path.join(OutputPath,"Realworld"+positionBiasSeverity+data_name_cur+metrics[1]+"ExploraDegree.pdf"), dpi=600, bbox_inches = 'tight', pad_inches = 0.05)
            plt.close(fig)