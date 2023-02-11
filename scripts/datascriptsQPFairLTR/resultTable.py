import sys
import os
import pandas as pd
import matplotlib.pyplot as plt 
sys.path.append("/home/taoyang/research/Tao_lib/BEL/src/BatchExpLaunch")
import results_org as results_org
# import BatchExpLaunch.s as tools
scriptPath=os.path.dirname(os.path.abspath(__file__))
os.chdir(scriptPath+"/../..")
def cal_timeFor1kLists(df):
    """
    This function return fairness for a dataframe
    """
    df["time1kLists"]=df["time"]/df["iterations"]*1000

step=19  
data_rename={            
            # "Movie":"Movie",\
            # "News":"News",\
            # "MSLR-WEB30k":"MSLR-WEB30k",\
            "MSLR-WEB10k":"MSLR10k",\
            # "Webscope_C14_Set1":"Webscope_C14_Set1",\
            # "istella-s":"istella-s"
            "MQ2008":"MQ2008",
            # "MQ2007":"MQ2007",
            "istella-s":"ist"
}
metric_name=['test_NDCG_1_aver','test_NDCG_3_aver','test_NDCG_5_aver',"test_disparity","time1kLists"]
metric_name=['test_NDCG_1_cumu','test_NDCG_3_cumu','test_NDCG_5_cumu',"test_disparity","time1kLists"]
# metric_name=["test_disparity","time1kLists"]
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
path_root="localOutput/Feb192022DataEstimatedAverage/"
path_root="localOutput/Feb222022Data/relvance_strategy_EstimatedAverage"
path_root="localOutput/Feb222022Data/relvance_strategy_TrueAverage"
path_root="localOutput/Apr252022LTR_small/relvance_strategy_TrueAverage"
path_root="localOutput/QPFairLTR/relvance_strategy_TrueAverage"
path_root="localOutput/QPFairLTRistella/relvance_strategy_TrueAverage"
path_root="localOutput/Apr30QPFairLTR/relvance_strategy_TrueAverage"
path_root="localOutput/July3QPFairLTR/relvance_strategy_TrueAverage"
path_root="localOutput/Jan252023QPFairLTRistella/relvance_strategy_TrueAverage"
# path_root="localOutput/July3QPFairLTRMSLR/relvance_strategy_TrueAverage"
# path_root="localOutput/Apr252022LTR_more/relvance_strategy_EstimatedAverage"

# path_root="localOutput/Mar292022Data20Docs/relvance_strategy_EstimatedAverage"
for positionBiasSeverity in positionBiasSeverities:
    result_list=[]
    OutputPath=os.path.join(path_root,"result")
    for metrics in metric_name:
        for datasets,data_name_cur in data_rename.items():
            result_validated={}
            datasets="dataset_name_"+datasets
            resultPath=os.path.join(path_root,positionBiasSeverity,datasets)
            
            if not os.path.isdir(resultPath):
                # print(path)       
                continue
            # result,result_mean=results_org.get_result_df(resultPath,groupby="iterations",rerun=True)
            result,result_mean=results_org.get_result_df(resultPath,groupby="iterations")
            results_org.iterate_applyfunction(result,cal_timeFor1kLists)
            # result,result_mean=results_org.get_result_df(resultPath,groupby="iterations")
            result_validated["Topk"]=result["fairness_strategy_Topk"]
            result_validated["Randomk"]=result["fairness_strategy_Randomk"]
            # result_validated["FairK"]=result["fairness_strategy_FairK"]
#             result_validated["ExploreK"]=result["fairness_strategy_ExploreK"]
            result_validated["FairCo"]=result["fairness_strategy_FairCo"]["fairness_tradeoff_param_1000"]["exploration_tradeoff_param_0.0"]
            result_validated["MMF"]=result["fairness_strategy_MMF"]["fairness_tradeoff_param_1.0"]["exploration_tradeoff_param_0.0"]
            result_validated["PLFair"]=result["fairness_strategy_PLFair"]["n_futureSession_10000000"]["fairness_tradeoff_param_1.0"]["exploration_tradeoff_param_0.0"]
            # result_validated["GradFair"]=result["fairness_strategy_GradFair"]["fairness_tradeoff_param_1000"]["exploration_tradeoff_param_0.0"]
            # result_validated["FairCo_explore-1"]=result["fairness_strategy_FairCo"]["fairness_tradeoff_param_1000"]["exploration_tradeoff_param_1"]
            # result_validated["FairCo_average_explore-1"]=result["fairness_strategy_FairCo_average"]["fairness_tradeoff_param_1000"]["exploration_tradeoff_param_1"]
            # result_validated["FairCo_explore-20"]=result["fairness_strategy_FairCo"]["fairness_tradeoff_param_1000"]["exploration_tradeoff_param_20"]
            # result_validated["FairCo_average_explore-20"]=result["fairness_strategy_FairCo_average"]["fairness_tradeoff_param_1000"]["exploration_tradeoff_param_20"]
            # result_validated["FairCo_explore-100-20"]=result["fairness_strategy_FairCo"]["fairness_tradeoff_param_100"]["exploration_tradeoff_param_20"]
            # result_validated["FairCo_average_explore-100-20"]=result["fairness_strategy_FairCo_average"]["fairness_tradeoff_param_100"]["exploration_tradeoff_param_20"]
            # result_validated["FairCo_explore-50-20"]=result["fairness_strategy_FairCo"]["fairness_tradeoff_param_50"]["exploration_tradeoff_param_20"]
            # result_validated["FairCo_average_explore-50-20"]=result["fairness_strategy_FairCo_average"]["fairness_tradeoff_param_50"]["exploration_tradeoff_param_20"]
            # result_validated["FairCo_explore-1-20"]=result["fairness_strategy_FairCo"]["fairness_tradeoff_param_1"]["exploration_tradeoff_param_20"]
            # result_validated["FairCo_average_explore-1-20"]=result["fairness_strategy_FairCo_average"]["fairness_tradeoff_param_1"]["exploration_tradeoff_param_20"]


            result_validated["LP"]=result["fairness_strategy_LP"]["n_futureSession_100000"]["fairness_tradeoff_param_1000"]["exploration_tradeoff_param_0.0"]
            # result_validated["LP_200"]=result["fairness_strategy_LP"]["n_futureSession_200"]["fairness_tradeoff_param_1000"]["exploration_tradeoff_param_0.0"]
            # result_validated["LP-1.0"]=result["fairness_strategy_LP"]["fairness_tradeoff_param_1.0"]["exploration_tradeoff_param_0.0"]
            # result_validated["LP-10"]=result["fairness_strategy_LP"]["fairness_tradeoff_param_10"]["exploration_tradeoff_param_0.0"]

            
            # result_validated["FairCo_multip."]=result["fairness_strategy_FairCo_multip."]["fairness_tradeoff_param_1000"]
            # result_validated["QPfair_2"]=result["fairness_strategy_QPfair"]["n_futureSession_2"]['fairness_tradeoff_param_1.0']
            # result_validated["QPfair_5"]=result["fairness_strategy_QPfair"]["n_futureSession_5"]['fairness_tradeoff_param_1.0']
            # result_validated["QPfairNDCG_20"]=result["fairness_strategy_QPfairNDCG"]["n_futureSession_20"]['fairness_tradeoff_param_1.0']
            # result_validated["QPfairNDCG_500"]=result["fairness_strategy_QPfairNDCG"]["n_futureSession_500"]['fairness_tradeoff_param_1.0']["exploration_tradeoff_param_10"]
            # result_validated["QPfairNDCG_500Hori"]=result["fairness_strategy_QPfairNDCGHorizontal"]["n_futureSession_500"]['fairness_tradeoff_param_1.0']["exploration_tradeoff_param_10"]

            # result_validated["QPfair_2"]=result["fairness_strategy_QPfair"]["n_futureSession_2"]['fairness_tradeoff_param_1.0']
            # result_validated["QPfair_5"]=result["fairness_strategy_QPfair"]["n_futureSession_5"]['fairness_tradeoff_param_1.0']
            result_validated["QPFair"]=result["fairness_strategy_QPFair"]["n_futureSession_100"]['fairness_tradeoff_param_1.0']["exploration_tradeoff_param_0.0"]
            result_validated["QPFair-Horiz."]=result["fairness_strategy_QPFair-Horiz."]["n_futureSession_100"]['fairness_tradeoff_param_1.0']["exploration_tradeoff_param_0.0"] 
            result_validated["ILP"]=result["fairness_strategy_ILP"]["fairness_tradeoff_param_1.0"]["exploration_tradeoff_param_0.0"]
            # result_validated["QPfair_100"]=result["fairness_strategy_QPfair"]["n_futureSession_100"]['fairness_tradeoff_param_1.0']
            # result_validated["QPfair_200"]=result["fairness_strategy_QPfair"]["n_futureSession_100"]['fairness_tradeoff_param_1.0']
            # result_validated["HQPfair_2"]=result["fairness_strategy_Hybrid"]["n_futureSession_2"]['fairness_tradeoff_param_1.0']
            # result_validated["HQPfair_20"]=result["fairness_strategy_Hybrid"]["n_futureSession_20"]['fairness_tradeoff_param_1.0']
            # result_validated["HQPfair_100"]=result["fairness_strategy_Hybrid"]["n_futureSession_100"]['fairness_tradeoff_param_1.0']
            result_vali_metrics=results_org.extract_step_metric(result_validated,metrics,step,data_name_cur+metrics)
            result_list=result_list+result_vali_metrics

    result_list=results_org.filteroutNone(result_list)
    result_dfram=pd.DataFrame(result_list, columns=["method","datasets","metrics"])
    result_dfram=result_dfram.pivot(index='method', columns='datasets', values='metrics')
    r,rstd=results_org.to_latex(result_dfram)
    os.makedirs(OutputPath, exist_ok=True)
    output_path=os.path.join(OutputPath,positionBiasSeverity+"mean_latex.csv")
    r.to_csv(output_path)
    output_path=os.path.join(OutputPath,positionBiasSeverity+"mean_std_latex.csv")
    rstd.to_csv(output_path)
    mean=results_org.to_mean(result_dfram)
    output_path=os.path.join(OutputPath,positionBiasSeverity+"mean.csv")
    mean.to_csv(output_path)
    for datasets,data_name_cur in data_rename.items():
        fig = plt.figure(figsize = (6, 3))
        methods=mean.index.tolist()
        if data_name_cur+"test_disparity" not in mean:
            continue
        unfairness=mean[data_name_cur+"test_disparity"].astype(float).tolist()
        # creating the bar plot
        plt.bar(methods, unfairness, 
                width = 0.4)
        plt.yscale("log")
        plt.xlabel("Methods")
        plt.ylabel("Unfairness")
        # plt.title("Students enrolled in different courses")
        fig.savefig(os.path.join(OutputPath,datasets+"FairnessCapacity.pdf"), dpi=300, bbox_inches = 'tight', pad_inches = 0.05)
        plt.close(fig)