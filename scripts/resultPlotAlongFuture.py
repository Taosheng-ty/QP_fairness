import sys
import os
import pandas as pd
import matplotlib.pyplot as plt 
sys.path.append("/home/taoyang/research/Tao_lib/BEL/src/BatchExpLaunch")
import results_org as results_org
scriptPath=os.path.dirname(os.path.abspath(__file__))
os.chdir(scriptPath+"/..")
path_root="localOutput/Feb142022"
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
result_list=[]

result_path=os.path.join(path_root,"result")
for datasets,data_name_cur in data_rename.items():
    fig, axs = plt.subplots(len(metric_name),figsize=(5,3*len(metric_name)), sharex=True)
    result_validated={}
    datasets="dataset_"+datasets
    path=os.path.join(path_root,datasets)

positionBiasSeverities=[
    "positionBiasSeverity_0",
    "positionBiasSeverity_1",
    "positionBiasSeverity_2"
    ]
result_list=[]
for positionBiasSeverity in positionBiasSeverities:
    OutputPath=os.path.join(path_root,"result")
    for datasets,data_name_cur in data_rename.items():
        fig, axs = plt.subplots(len(metric_name),figsize=(5,3*len(metric_name)), sharex=True)
        datasets="dataset_"+datasets
        resultPath=os.path.join(path_root,positionBiasSeverity,datasets)
        result_validated={}
        if not os.path.isdir(resultPath):
            # print(path)       
            continue
        result,result_mean=results_org.get_result_df(resultPath,groupby="iterations")
        result_validated["QP"]=results_org.getGrandchildNode(result["fairness_strategy_QPfair"],"fairness_tradeoff_param_1.0")

        for ind,metrics in enumerate(metric_name):
            results_org.paramIterationPlot(result_validated, metrics,ax=axs[ind],step=step)
            axs[ind].set_ylabel(metrics)
            axs[ind].set_xlabel("# of future sessions to consider")
            axs[ind].set_title(data_name_cur)
            # axs[ind].set_xscale("log")
            # axs[0].set_yscale("symlog")
            axs[ind].legend(bbox_to_anchor=(1.1, 1.05))    
        
        fig.savefig(os.path.join(OutputPath,positionBiasSeverity+data_name_cur+"performance_along_future.pdf"), dpi=600, bbox_inches = 'tight', pad_inches = 0.05)
        plt.close(fig)