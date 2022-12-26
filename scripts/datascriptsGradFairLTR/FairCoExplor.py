import sys
import os
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib
plt.rcParams['pdf.fonttype']=42
font = {'size'   : 12}

matplotlib.rc('font', **font)
sys.path.append("/home/taoyang/research/Tao_lib/BEL/src/BatchExpLaunch")
import config
from matplotlib import scale
scale.register_scale(config.Mylog2f)
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
path_root="localOutput/Apr252022LTR_more/relvance_strategy_EstimatedAverage"
path_root="localOutput/Apr262022LTR/relvance_strategy_EstimatedAverage"
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
metric_name=[["test_disparity",'test_NDCG_1_aver'],["test_disparity",'test_NDCG_3_aver'],["test_disparity",'test_NDCG_5_aver']]
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
    OutputPath=os.path.join(path_root,"result","MCFair")
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
        # result_validated["GradFair"]=result["fairness_strategy_FairCo_average"]
        # result_validated["LP_1"]=result["fairness_strategy_LP"]["n_futureSession_1"]
        # result_validated["LP"]=result["fairness_strategy_LP"]["n_futureSession_1"]


        # result_validated["ILP"]=result["fairness_strategy_ILP"]
        # for method in result_validated:
        #     result_validated[method]=results_org.getGrandchildNode(result_validated[method],"exploration_tradeoff_param_0.0")
        # result_validated["QPfairNDCG_500"]=result["fairness_strategy_QPfairNDCG"]["n_futureSession_500"]
        # result_validated["QPfairNDCG_500Hori"]=result["fairness_strategy_QPfairNDCGHorizontal"]["n_futureSession_500"]
        result_validated["MCFair(Ours)"]=results_org.getGrandchildNode(result["fairness_strategy_GradFair"],"exploration_tradeoff_param_100")
        result_validated["FairCo"]=results_org.getGrandchildNode(result["fairness_strategy_FairCo"],"exploration_tradeoff_param_0.0")
        result_validated["FairCo w/ Explor."]=results_org.getGrandchildNode(result["fairness_strategy_FairCo"],"exploration_tradeoff_param_100")
        

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
        # result_validatedScatter["Topk"]=result["fairness_strategy_Topk"]
        # result_validatedScatter["RandomK"]=result["fairness_strategy_Randomk"]
        # result_validatedScatter["FairK"]=result["fairness_strategy_FairK"]
        # result_validatedScatter["ExploreK"]=result["fairness_strategy_ExploreK"]
        config.desiredGradFairMarker["FairCo w/ Explor."]="^"
        for ind,metrics in enumerate(metric_name):
            fig, axs = plt.subplots(figsize=(6.4,2.4))
            results_org.RequirementPlot(result_validated, metrics,\
                                        desiredColorDict=config.desiredGradFairColor,desiredMarkerDict=config.desiredGradFairMarker,ax=axs,step=step)
            
            # for line in axs.lines:
            #     line.set_marker(None)
            # results_org.TradeoffScatter(result_validatedScatter, metrics,ax=axs[ind],step=step)
            axs.set_ylabel(metric_name_dict[metrics[1]])
            axs.set_xlabel(metric_name_dict[metrics[0]])
            # axs.set_title(data_name_cur)
            # axs[ind].set_xscale("log")
            # axs[0].set_yscale("symlog")
            # axs[ind].legend(bbox_to_anchor=(1.1, 1.05))    
            axs.set_xscale("function",functions=xscaleFcn[data_name_cur]) 
            axs.set_yscale("function",functions=yscaleFcn[data_name_cur]) 
            axs.set_xticks(ticks=[3000,20000,40000])
            # plt.locator_params(axis='x', nbins=4)
            plt.locator_params(axis='y', nbins=4)
            axs.legend(framealpha=0.2,bbox_to_anchor=(1.04,1), loc="upper left")
            fig.savefig(os.path.join(OutputPath,metrics[1]+positionBiasSeverity+data_name_cur+"FairCoExploration.pdf"), dpi=600, bbox_inches = 'tight', pad_inches = 0.05)
            plt.close(fig)
            