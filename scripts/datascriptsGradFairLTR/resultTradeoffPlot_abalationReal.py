import sys
import os
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib
# plt.rcParams['pdf.fonttype']=42
matplotlib.rcParams['text.usetex'] = True
font = {'size'   : 12}
import config
sys.path.append("/home/taoyang/research/Tao_lib/BEL/src/BatchExpLaunch")
import results_org as results_org
results_org.figureConfig()
scriptPath=os.path.dirname(os.path.abspath(__file__))
os.chdir(scriptPath+"/../..")
path_root="localOutput/Feb192022DataTrueAver/"
# path_root="localOutput/Feb182022Data/"
# path_root="localOutput/Feb192022DataEstimatedAverage/"
path_root="localOutput/Feb222022Data/relvance_strategy_EstimatedAverage"
# path_root="localOutput/Feb222022Data/relvance_strategy_TrueAverage"
path_root="localOutput/Mar292022Data20Docs/relvance_strategy_TrueAverage"
path_root="localOutput/Apr262022LTR/relvance_strategy_EstimatedAverage"
# path_root="localOutput/Apr30QPFairLTR/relvance_strategy_EstimatedAverage"

step=19  
data_rename={            
            # "Movie":"Movie",\
            # "News":"News",\
            # "MSLR-WEB30k":"MSLR-WEB30k",\
            # "MSLR-WEB10k":"MSLR-WEB10k",\
            # "Webscope_C14_Set1":"Webscope_C14_Set1",\
            # "istella-s":"ist",
            "MQ2008":"MQ2008",
            # "MQ2007":"MQ2007",
}
metric_name=[["test_disparity",'test_NDCG_1_aver'],["test_disparity",'test_NDCG_3_aver'],\
    ["test_disparity",'test_NDCG_5_aver'],["test_disparity",'test_NDCG_1_cumu'],["test_disparity",'test_NDCG_3_cumu'],\
    ["test_disparity",'test_NDCG_5_cumu'],]
# metric_name=[["disparity",'NDCG_3_aver'],["disparity",'NDCG_5_aver']]

metric_name_dict={"test_NDCG_1_aver":"NDCG@1","test_NDCG_3_aver":"NDCG@3","test_NDCG_5_aver":"NDCG@5",\
    "test_NDCG_1_cumu":"cNDCG@1","test_NDCG_3_cumu":"cNDCG@3","test_NDCG_5_cumu":"cNDCG@5","test_disparity":"Unfairness tolerance"}
result_list=[]
same=[lambda x:x, lambda x:x]
yMQfunctions=results_org.setScaleFunction(a=201,b=1,low=False)
yIsfunctions=results_org.setScaleFunction(a=201,b=1,low=False)
yscaleFcn={"MQ2008":yMQfunctions,"ist":same}
xMQfunctions=results_org.setScaleFunction(a=100,b=1,low=True)
xIsfunctions=results_org.setScaleFunction(a=10,b=1,low=True)
xscaleFcn={"MQ2008":xMQfunctions,"ist":same}

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
    OutputPath=os.path.join(path_root,"result","GradFair")
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
        result_validatedMC={}
        # result_validated["Eff.+Uncert.+Fair."]= results_org.getGrandchildNode(result["fairness_strategy_GradFair"],"exploration_tradeoff_param_100")
        result_validated["Eff.+Uncert.+Fair."]= results_org.getGrandchildNode(result["fairness_strategy_GradFair"],"exploration_tradeoff_param_100")

        result_validated["Eff.+Fair."]= results_org.getGrandchildNode(result["fairness_strategy_GradFair"],"exploration_tradeoff_param_0.0")
        result_validated["Eff.+Uncert."]= result["fairness_strategy_GradFair"]["fairness_tradeoff_param_0.0"]
        result_validated["Fair.+Uncert."]= result["fairness_strategy_onlyFairness"]["fairness_tradeoff_param_1"]

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
            fig, axs = plt.subplots(figsize=(6.4,2.4))
            # results_org.RequirementPlot(result_validatedMC, metrics,ax=axs,step=step)
            results_org.TradeoffPlot(result_validated, metrics,ax=axs,step=step)
            for line in axs.lines:
                line.set_marker(None)
                # line.set_markersize(10)
            results_org.TradeoffScatter(result_validatedScatter, metrics,ax=axs,step=step)
            axs.set_ylabel(metric_name_dict[metrics[1]])
            axs.set_xlabel(metric_name_dict[metrics[0]])
            axs.set_xscale("function",functions=xscaleFcn[data_name_cur]) 
            axs.set_yscale("function",functions=yscaleFcn[data_name_cur]) 
            # plt.locator_params(axis='x', nbins=3)
            axs.set_xticks(ticks=[3000,20000,40000,60000])
            plt.locator_params(axis='y', nbins=4)   
            axs.legend(framealpha=0.2,bbox_to_anchor=(1.04,1), loc="upper left")
            fig.savefig(os.path.join(OutputPath,metrics[1]+positionBiasSeverity+data_name_cur+"Ablation_tradeoffplot.pdf"), dpi=600, bbox_inches = 'tight', pad_inches = 0.05)
            plt.close(fig)