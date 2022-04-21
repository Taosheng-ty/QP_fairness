import sys
import os
import pandas as pd
sys.path.append("/home/taoyang/research/Tao_lib/BEL/src/BatchExpLaunch")
import results_org as results_org
scriptPath=os.path.dirname(os.path.abspath(__file__))
os.chdir(scriptPath+"/..")
path_root="localOutput/Feb142022/"
step=19  
data_rename={            
            "MQ2007":"MQ2007",\
            "MQ2008":"MQ2008",\
            # "MSLR-WEB30k":"MSLR-WEB30k",\
            # "MSLR-WEB10k":"MSLR-WEB10k",\
            # "Webscope_C14_Set1":"Webscope_C14_Set1",\
            # "istella-s":"istella-s"
}
metric_name=['test_NDCG_1_aver','test_NDCG_3_aver','test_NDCG_5_aver',"test_disparity",]
metric_name_dict={"discounted_sum_test_ndcg":"Cum-NDCG","test_fairness":"bfairness","average_sum_test_ndcg":"average_cum_ndcg",\
    'f1_test_rel_fair':'crf-f1',"neg_test_exposure_disparity_not_divide_qfreq":"cnegdisparity",\
        'test_exposure_disparity_not_divide_qfreq':"Disparity"}
positionBiasSeverities=[
    "positionBiasSeverity_0",
    "positionBiasSeverity_1",
    "positionBiasSeverity_2"
    ]

for positionBiasSeverity in positionBiasSeverities:
    result_list=[]
    OutputPath=os.path.join(path_root,"result")
    for metrics in metric_name:
        for datasets,data_name_cur in data_rename.items():
            result_validated={}
            datasets="dataset_"+datasets
            resultPath=os.path.join(path_root,positionBiasSeverity,datasets)
            
            if not os.path.isdir(resultPath):
                # print(path)       
                continue
            result,result_mean=results_org.get_result_df(resultPath,groupby="iterations",rerun=True)
            result_validated["Topk"]=result["fairness_strategy_Topk"]
            result_validated["Randomk"]=result["fairness_strategy_Randomk"]
            result_validated["FairCo"]=result["fairness_strategy_FairCo"]["fairness_tradeoff_param_100000"]
            result_validated["FairCo_multip."]=result["fairness_strategy_FairCo_multip."]["fairness_tradeoff_param_100000"]
            result_validated["FairCo_average"]=result["fairness_strategy_FairCo_average"]["fairness_tradeoff_param_100000"]
            # result_validated["QPfair_5"]=result["fairness_strategy_QPfair"]["n_futureSession_5"]['fairness_tradeoff_param_1.0']
            result_validated["QPfair"]=result["fairness_strategy_QPfair"]["n_futureSession_20"]['fairness_tradeoff_param_1.0']
            # result_validated["QPfair_100"]=result["fairness_strategy_QPfair"]["n_futureSession_100"]['fairness_tradeoff_param_1.0']
            result_vali_metrics=results_org.extract_step_metric(result_validated,metrics,step,data_name_cur+metrics[5:])
            result_list=result_list+result_vali_metrics
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