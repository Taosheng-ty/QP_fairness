import sys
import os
import pandas as pd
sys.path.append("/home/taoyang/research/Tao_lib/BEL/src/BatchExpLaunch")
# sys.path.append("/raid/taoyang/miniconda3/pkgs/texlive-core-20180414-pl526hc0a3334_3/share/texmf-dist/web2c")
import results_org as results_org
import config
import matplotlib.pyplot as plt 
import matplotlib
# plt.rcParams['pdf.fonttype']=42
matplotlib.rcParams['text.usetex'] = True
font = {'size'   : 12}
from matplotlib import scale

matplotlib.rc('font', **font)
# import BatchExpLaunch.s as tools
scriptPath=os.path.dirname(os.path.abspath(__file__))
os.chdir(scriptPath+"/../..")

step=19  
data_rename={            
            # "Movie":"Movie",\
            # "News":"News",\
            # "MSLR-WEB30k":"MSLR-WEB30k",\
            # "MSLR-WEB10k":"MSLR-WEB10k",\
            # "Webscope_C14_Set1":"Webscope_C14_Set1",\
            # "istella-s":"istella-s"，
            "MQ2008":"MQ2008",
            # "MQ2007":"MQ2007",
            "istella-s":"ist"
}
MQfunctions=results_org.setScaleFunction(a=210,b=1,low=False)
Isfunctions=results_org.setScaleFunction(a=-10,b=1,low=True)
scaleFcn={"MQ2008":MQfunctions,"ist":Isfunctions}
xMQfunctions=results_org.setScaleFunction(a=0,b=1,low=True)
# xIsfunctions=results_org.setScaleFunction(a=10,b=1,low=True)
xIsfunctions=[lambda x:x, lambda x:x]
xscaleFcn={"MQ2008":xMQfunctions,"ist":xIsfunctions}

metric_name=['test_NDCG_1_aver','test_NDCG_3_aver','test_NDCG_5_aver']
metric_name=['test_NDCG_1_cumu','test_NDCG_2_cumu','test_NDCG_3_cumu','test_NDCG_4_cumu','test_NDCG_5_cumu']
x=[1,2,3,4,5]
metric_name_dict={"discounted_sum_test_ndcg":"Cum-NDCG","test_fairness":"bfairness","average_sum_test_ndcg":"average_cum_ndcg",\
    'f1_test_rel_fair':'crf-f1',"neg_test_exposure_disparity_not_divide_qfreq":"cnegdisparity",\
        'test_exposure_disparity_not_divide_qfreq':"Disparity"}
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
path_root="localOutput/Apr262022LTR/relvance_strategy_TrueAverage"
# path_root="localOutput/Apr30QPFairLTR/relvance_strategy_TrueAverage"
ytop={"ist":210,"MQ2008":201}
# path_root="localOutput/Mar292022Data20Docs/relvance_strategy_EstimatedAverage"
for positionBiasSeverity in positionBiasSeverities:
    
    OutputPath=os.path.join(path_root,"result","MCFair")
    for datasets,data_name_cur in data_rename.items():
        fig, ax = plt.subplots(figsize=(6.0,1.8))
        result_list=[]  
        result_validated={}
        datasets="dataset_name_"+datasets
        resultPath=os.path.join(path_root,positionBiasSeverity,datasets)
        
        if not os.path.isdir(resultPath):
            # print(path)       
            continue
        
        # result,result_mean=results_org.get_result_df(resultPath,groupby="iterations")
        _,result=results_org.get_result_df(resultPath,groupby="iterations")
        # result,result_mean=results_org.get_result_df(resultPath,groupby="iterations")
        result_validated["FairCo"]=result["fairness_strategy_FairCo"]["fairness_tradeoff_param_1000"]["exploration_tradeoff_param_0.0"]
        result_validated["MMF"]=result["fairness_strategy_MMF"]["fairness_tradeoff_param_1.0"]["exploration_tradeoff_param_0.0"]
        # result_validated["GradFair"]=result["fairness_strategy_FairCo_average"]["fairness_tradeoff_param_1000"]["exploration_tradeoff_param_0.0"]



        # result_validated["GradFair"]=result["fairness_strategy_FairCo_average"]["fairness_tradeoff_param_1"]["exploration_tradeoff_param_10"]
        result_validated["TopK"]=result["fairness_strategy_Topk"]
        # result_validated["RandomK"]=result["fairness_strategy_Randomk"]
        result_validated["FairK(Ours)"]=result["fairness_strategy_FairK"]
        result_validated["ExploreK"]=result["fairness_strategy_ExploreK"]
        result_validated["MCFair(Ours)"]=result["fairness_strategy_GradFair"]["fairness_tradeoff_param_1000"]["exploration_tradeoff_param_0.0"]
        if "fairness_strategy_LP" in result:
            result_validated["ILP"]=result["fairness_strategy_ILP"]["fairness_tradeoff_param_1.0"]["exploration_tradeoff_param_0.0"]
            result_validated["LP"]=result["fairness_strategy_LP"]["n_futureSession_100000"]["fairness_tradeoff_param_1000"]["exploration_tradeoff_param_0.0"]
        result_validated["PLFair"]=result["fairness_strategy_PLFair"]["n_futureSession_10000000"]["fairness_tradeoff_param_1.0"]["exploration_tradeoff_param_0.0"]
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
        results_org.plot(result_dict,ax=ax,\
                                        desiredColorDict=config.desiredGradFairColor,desiredMarkerDict=config.desiredGradFairMarker)
        ax.set_xticks(x)
        ax.set_xlabel("Cutoff")
        ax.set_ylabel("cNDCG")
        ax.set_ylim(top=ytop[data_name_cur])
        # ax.set_yscale("log")
        ax.set_yscale("function",functions=scaleFcn[data_name_cur])
        ax.set_xscale("function",functions=xscaleFcn[data_name_cur])
        # ax.legend(bbox_to_anchor=(1.1, 1.05))
        legend,handles,labels=results_org.reorderLegend(config.desiredGradFair,ax,returnHandles=True)
        # legend.nrow=2
        ax.legend(handles,labels,bbox_to_anchor=(1.04,1), loc="upper left",ncol=2)
        plt.locator_params(axis='x', nbins=5)
        os.makedirs(OutputPath, exist_ok=True)
        fig.savefig(os.path.join(OutputPath,positionBiasSeverity+data_name_cur+"NDCGcutoffcumu.pdf"), dpi=600, bbox_inches = 'tight', pad_inches = 0.05)
        plt.close(fig)
