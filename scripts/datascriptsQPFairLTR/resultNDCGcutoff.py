import sys
import os
import pandas as pd
sys.path.append("/home/taoyang/research/Tao_lib/BEL/src/BatchExpLaunch")
import results_org as results_org
# import BatchExpLaunch.results_org as results_org
import config
import matplotlib.pyplot as plt 
import matplotlib
import numpy as np
plt.rcParams['pdf.fonttype']=42
font = {'size'   : 16}
matplotlib.rc('font', **font)
# import BatchExpLaunch.s as tools
scriptPath=os.path.dirname(os.path.abspath(__file__))
os.chdir(scriptPath+"/../..")


# metric_name=['test_NDCG_1_aver','test_NDCG_2_aver','test_NDCG_3_aver','test_NDCG_4_aver','test_NDCG_5_aver']
metric_name=['test_NDCG_1_cumu','test_NDCG_2_cumu','test_NDCG_3_cumu','test_NDCG_4_cumu','test_NDCG_5_cumu']
x=[1,2,3,4,5]

metric_name_dict={"test_NDCG_1_aver":"NDCG@1","test_NDCG_2_aver":"NDCG@2",\
                  "test_NDCG_3_aver":"NDCG@3","test_NDCG_4_aver":"NDCG@4","test_NDCG_5_aver":"NDCG@5",\
    "test_NDCG_1_cumu":"cNDCG@1","test_NDCG_3_cumu":"cNDCG@3","test_NDCG_5_cumu":"cNDCG@5",}
positionBiasSeverities=[
    # "positionBiasSeverity_0",
    "positionBiasSeverity_1",
    # "positionBiasSeverity_2"
    ]
path_root="localOutput/Feb182022Data/"
path_root="localOutput/Feb192022DataTrueAver/"
# path_root="localOutput/Feb192022DataEstimatedAverage/"
# path_root="localOutput/Feb222022Data/relvance_strategy_EstimatedAverage"
path_root="localOutput/Feb222022Data/relvance_strategy_TrueAverage"
path_root="localOutput/Apr252022LTR_more/relvance_strategy_TrueAverage"
path_root="localOutput/QPFairLTR/relvance_strategy_TrueAverage"
path_root="localOutput/QPFairLTRistella/relvance_strategy_TrueAverage"
path_root="localOutput/Apr30QPFairLTR/relvance_strategy_TrueAverage"
path_root="localOutput/July3QPFairLTR/relvance_strategy_TrueAverage"
path_root="localOutput/July3QPFairLTR/relvance_strategy_TrueAverage"
# path_root="localOutput/July3QPFairLTRMSLR/relvance_strategy_TrueAverage"
step=19  

# xMQfunctions=results_org.setScaleFunction(a=203,b=1,low=False)
Equal=[lambda x:x, lambda x:x]
MQfunctions=results_org.setScaleFunction(a=203,b=1,low=False)
MSLR10kfunctions=results_org.setScaleFunction(a=50,b=1,low=True)
yscaleFcn={"MQ2008":MQfunctions,"MSLR10k":MSLR10kfunctions}
# xIsfunctions=[lambda x: np.log(np.log(210-x)), lambda x:210-np.exp(np.exp(x))]
# yscaleFcn={"MQ2008":yMQfunctions,"MSLR10k":yIsfunctions}
# xscaleFcn={"MQ2008":xMQfunctions,"MSLR10k":xIsfunctions}
# yscaleFcn={"MQ2008":xMQfunctions,"MSLR10k":xIsfunctions}
xscaleFcn={"MQ2008":Equal,"MSLR10k":Equal}
y_lim={"MQ2008":201,"MSLR10k":230}
data_rename={            
            # "Movie":"Movie",\
            # "News":"News",\
            # "MSLR-WEB30k":"MSLR-WEB30k",\
            "MSLR-WEB10k":"MSLR10k",\
            # "Webscope_C14_Set1":"Webscope_C14_Set1",\
            # "istella-s":"istella-s"，
            "MQ2008":"MQ2008",
            # "MQ2007":"MQ2007",
            # "istella-s":"ist",
}
# path_root="localOutput/Mar292022Data20Docs/relvance_strategy_EstimatedAverage"
yticks={"MQ2008":[100,170,200],"MSLR10k":[70,100,200]}
for positionBiasSeverity in positionBiasSeverities:
    
    OutputPath=os.path.join(path_root,"result")
    for datasets,data_name_cur in data_rename.items():
#         fig, ax = plt.subplots(figsize=(6.4,2.4))
        fig, ax = plt.subplots()
        result_list=[]  
        result_validated={}
        datasets="dataset_name_"+datasets
        resultPath=os.path.join(path_root,positionBiasSeverity,datasets)
        
        if not os.path.isdir(resultPath):
            # print(path)       
            continue
        
        result,result_mean=results_org.get_result_df(resultPath,groupby="iterations")
        # result,result_mean=results_org.get_result_df(resultPath,groupby="iterations")
        result_validated["TopK"]=result["fairness_strategy_Topk"]
        result_validated["RandomK"]=result["fairness_strategy_Randomk"]

        result_validated["FairCo"]=result["fairness_strategy_FairCo"]["fairness_tradeoff_param_1000"]["exploration_tradeoff_param_0.0"]
        result_validated["MMF"]=result["fairness_strategy_MMF"]["fairness_tradeoff_param_1.0"]["exploration_tradeoff_param_0.0"]

        # result_validated["GradFair"]=result["fairness_strategy_FairCo_average"]["fairness_tradeoff_param_1000"]["exploration_tradeoff_param_0.0"]
        if "fairness_strategy_LP" in result:
            result_validated["ILP"]=result["fairness_strategy_ILP"]["fairness_tradeoff_param_1.0"]["exploration_tradeoff_param_0.0"]
            result_validated["LP"]=result["fairness_strategy_LP"]["n_futureSession_100000"]["fairness_tradeoff_param_1000"]["exploration_tradeoff_param_0.0"]
        # result_validated["GradFair(Ours)"]=result["fairness_strategy_GradFair"]["fairness_tradeoff_param_1000"]["exploration_tradeoff_param_0.0"]
        result_validated["PLFair"]=result["fairness_strategy_PLFair"]["n_futureSession_10000000"]["fairness_tradeoff_param_1.0"]["exploration_tradeoff_param_0.0"]
        # result_validated["FairK(Ours)"]=result["fairness_strategy_FairK"]
        result_validated["FARA(Ours)"]=result["fairness_strategy_QPFair"]["n_futureSession_100"]['fairness_tradeoff_param_1.0']["exploration_tradeoff_param_0.0"]
        result_validated["FARA-Horiz.(Ours)"]=result["fairness_strategy_QPFair-Horiz."]["n_futureSession_100"]['fairness_tradeoff_param_1.0']["exploration_tradeoff_param_0.0"] 
        # result_validated["GradFair"]=result["fairness_strategy_FairCo_average"]["fairness_tradeoff_param_1"]["exploration_tradeoff_param_10"]


        
        # result_validated["ExploreK"]=result["fairness_strategy_ExploreK"]
        # result_validated["QPfairNDCG_500"]=result["fairness_strategy_QPfairNDCG"]["n_futureSession_500"]['fairness_tradeoff_param_1.0']
        # result_validated["QPfairNDCG_500Hori"]=result["fairness_strategy_QPfairNDCGHorizontal"]["n_futureSession_500"]['fairness_tradeoff_param_1.0']       
        # result_validated["FairCo_multip."]=result["fairness_strategy_FairCo_multip."]["fairness_tradeoff_param_1000"]
        # result_validated["FairCo_average"]=result["fairness_strategy_FairCo_average"]["fairness_tradeoff_param_1000"]
        # result_validated["QPfair_2"]=result["fairness_strategy_QPfair"]["n_futureSession_2"]['fairness_tradeoff_param_1.0']
        # result_validated["QPfairQuota"]=result["fairness_strategy_QPfair"]["n_futureSession_200"]['fairness_tradeoff_param_1.0']
        # result_validated["QPfairNDCG_20"]=result["fairness_strategy_QPfairNDCG"]["n_futureSession_20"]['fairness_tradeoff_param_1.0']
        # result_validated["QPfairNDCG"]=result["fairness_strategy_QPfairNDCG"]["n_futureSession_200"]['fairness_tradeoff_param_1.0']
        # result_validated["QPfair_2"]=result["fairness_strategy_QPfair"]["n_futureSession_2"]['fairness_tradeoff_param_1.0']
        # result_validated["QPfair_5"]=result["fairness_strategy_QPfair"]["n_futureSession_5"]['fairness_tradeoff_param_1.0']
        # result_validated["QPfair_20"]=result["fairness_strategy_QPfair"]["n_futureSession_20"]['fairness_tradeoff_param_1.0']
        # result_validated["QPfair_100"]=result["fairness_strategy_QPfair"]["n_futureSession_100"]['fairness_tradeoff_param_1.0']
        # result_validated["QPfair_200"]=result["fairness_strategy_QPfair"]["n_futureSession_100"]['fairness_tradeoff_param_1.0']
        # result_validated["HQPfair_2"]=result["fairness_strategy_Hybrid"]["n_futureSession_2"]['fairness_tradeoff_param_1.0']
        # result_validated["HQPfair_20"]=result["fairness_strategy_Hybrid"]["n_futureSession_20"]['fairness_tradeoff_param_1.0']
        # result_validated["HQPfair_100"]=result["fairness_strategy_Hybrid"]["n_futureSession_100"]['fairness_tradeoff_param_1.0']
        result_validated=results_org.reorderDict(result_validated,config.desiredGradFair)
        for metrics in metric_name:
            result_vali_metrics=results_org.extract_step_metric(result_validated,metrics,step,metrics)
            result_list=result_list+result_vali_metrics
        result_list=results_org.filteroutNone(result_list)
        result_dfram=pd.DataFrame(result_list, columns=["method","metrics","metricsValue"])        
        result_dfram=result_dfram.pivot(index='method', columns='metrics', values='metricsValue')
        result_dfram=result_dfram.reindex(columns=metric_name)
        
        result_dict={index:[x,result_dfram.loc[index].to_list()] for index in result_validated.keys()}
        results_org.plot(result_dict,ax=ax,desiredColorDict=config.desiredGradFairColor,\
                                            desiredMarkerDict=config.desiredGradFairMarker,errbar=False)
        ax.set_xticks(x)
        ax.set_xlabel("Cutoff, $k_c$")
        ax.set_ylabel("cNDCG@$k_c$")
        # ax.set_yscale("log")
        # ax.set_xscale("function",functions=xscaleFcn[data_name_cur]) 
        ax.set_yscale("function",functions=yscaleFcn[data_name_cur])       
        ax.set_ylim(65,y_lim[data_name_cur])
        legend,handles,labels=results_org.reorderLegend(config.desiredGradFair,ax,returnHandles=True)
        # legend.nrow=2
        # ax.legend(handles,labels,bbox_to_anchor=(1, 0.5), loc="center left")
        legend = ax.legend(handles, labels, loc=3,ncol=5, framealpha=1, frameon=True,bbox_to_anchor=(1.1, 1.05),columnspacing=0.5)
        results_org.export_legend(legend,OutputPath+'legendNDCGcutoff.pdf')
        legend.remove()
        ax.set_yticks(ticks=yticks[data_name_cur])
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # ax.legend()
        os.makedirs(OutputPath, exist_ok=True)
        fig.savefig(os.path.join(OutputPath,positionBiasSeverity+data_name_cur+"NDCGcutoffcumu.pdf"), dpi=300, bbox_inches = 'tight', pad_inches = 0.05)
        plt.close(fig)
