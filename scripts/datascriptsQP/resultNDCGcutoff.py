import sys
import os
import pandas as pd
sys.path.append("/home/taoyang/research/Tao_lib/BEL/src/BatchExpLaunch")
import results_org as results_org
import config
import matplotlib.pyplot as plt 
# import BatchExpLaunch.s as tools
scriptPath=os.path.dirname(os.path.abspath(__file__))
os.chdir(scriptPath+"/../..")

step=19  
data_rename={            
            "Movie":"Movie",\
            # "News":"News",\
            # "MSLR-WEB30k":"MSLR-WEB30k",\
            # "MSLR-WEB10k":"MSLR-WEB10k",\
            # "Webscope_C14_Set1":"Webscope_C14_Set1",\
            # "istella-s":"istella-s"
}
metric_name=['NDCG_1_aver','NDCG_3_aver','NDCG_5_aver']
x=[1,3,5]
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
path_root="localOutput/Mar292022Data20Docs/relvance_strategy_TrueAverage"
# path_root="localOutput/Mar292022Data20Docs/relvance_strategy_EstimatedAverage"

for positionBiasSeverity in positionBiasSeverities:
    
    OutputPath=os.path.join(path_root,"result")
    for datasets,data_name_cur in data_rename.items():
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
        result_validated["FairCo"]=result["fairness_strategy_FairCo"]["fairness_tradeoff_param_1000"]["exploration_tradeoff_param_0.0"]
        # result_validated["GradFair"]=result["fairness_strategy_FairCo_average"]["fairness_tradeoff_param_1000"]["exploration_tradeoff_param_0.0"]
        result_validated["ILP"]=result["fairness_strategy_ILP"]["fairness_tradeoff_param_1.0"]["exploration_tradeoff_param_0.0"]
        result_validated["LP"]=result["fairness_strategy_LP"]["fairness_tradeoff_param_1000"]["exploration_tradeoff_param_0.0"]
        # result_validated["GradFair"]=result["fairness_strategy_FairCo_average"]["fairness_tradeoff_param_1"]["exploration_tradeoff_param_10"]
        result_validated["Topk"]=result["fairness_strategy_Topk"]
        result_validated["RandomK"]=result["fairness_strategy_Randomk"]
        # result_validated["FairK"]=result["fairness_strategy_FairK"]
        # result_validated["ExploreK"]=result["fairness_strategy_ExploreK"]
        result_validated["QPFair (Ours)"]=result["fairness_strategy_QPfairNDCG"]["n_futureSession_200"]['fairness_tradeoff_param_1.0']
        result_validated["QPFair-Horiz"]=result["fairness_strategy_QPfairNDCGHorizontal"]["n_futureSession_200"]['fairness_tradeoff_param_1.0']       
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
        for metrics in metric_name:
            result_vali_metrics=results_org.extract_step_metric(result_validated,metrics,step,metrics)
            result_list=result_list+result_vali_metrics
        result_list=results_org.filteroutNone(result_list)
        result_dfram=pd.DataFrame(result_list, columns=["method","metrics","metricsValue"])        
        result_dfram=result_dfram.pivot(index='method', columns='metrics', values='metricsValue')
        result_dfram=result_dfram.reindex(columns=metric_name)
        
        result_dict={index:[x,result_dfram.loc[index].to_list()] for index in result_dfram.index}
        results_org.plot(result_dict,ax=ax)
        ax.set_xticks(x)
        ax.set_xlabel("Cutoff")
        ax.set_ylabel("NDCG")
        # ax.legend(bbox_to_anchor=(1.1, 1.05))  
        legend=results_org.reorderLegend(config.desiredQPFair,ax)
        legend.get_frame().set_facecolor((1, 1, 1, 0.1))
        # ax.legend(bbox_to_anchor=(1.1, 1.05))  
        os.makedirs(OutputPath, exist_ok=True)
        fig.savefig(os.path.join(OutputPath,positionBiasSeverity+data_name_cur+"NDCGcutoffQP.pdf"), dpi=600, bbox_inches = 'tight', pad_inches = 0.05)
        plt.close(fig)
