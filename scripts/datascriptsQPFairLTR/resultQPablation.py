import sys
import os
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.ticker as mticker
import matplotlib
font = {'family' : 'normal',
        'size'   : 21}

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
path_root="localOutput/QPFairLTR/relvance_strategy_EstimatedAverage"
path_root="localOutput/QPFairLTRistella/relvance_strategy_EstimatedAverage"
path_root="localOutput/Apr30QPFairLTR/relvance_strategy_EstimatedAverage"
path_root="localOutput/July3QPFairLTR/relvance_strategy_EstimatedAverage"
path_root="localOutput/July3QPFairLTRMSLR/relvance_strategy_EstimatedAverage"
# path_root="localOutput/Jan252023QPFairLTRistella/relvance_strategy_EstimatedAverage"
step=19  
data_rename={            
            # "Movie":"Movie",\
            # "News":"News",\
            # "MSLR-WEB30k":"MSLR-WEB30k",\
            "MSLR-WEB10k":"MSLR10k",\
            # "Webscope_C14_Set1":"Webscope_C14_Set1",\
            # "istella-s":"istella-s"，
            "MQ2008":"MQ2008",
            # "MQ2007":"MQ2007",
            "istella-s":"istella-s",
}
metric_name=[["test_disparity",'test_NDCG_1_aver'],["test_disparity",'test_NDCG_3_aver'],\
    ["test_disparity",'test_NDCG_5_aver'],["test_disparity",'test_NDCG_1_cumu'],["test_disparity",'test_NDCG_3_cumu'],\
    ["test_disparity",'test_NDCG_5_cumu'],]
# metric_name=[["disparity",'NDCG_3_aver'],["disparity",'NDCG_5_aver']]

metric_name_dict={"test_NDCG_1_aver":"NDCG@1","test_NDCG_3_aver":"NDCG@3","test_NDCG_5_aver":"NDCG@5",\
    "test_NDCG_1_cumu":"cNDCG","test_NDCG_3_cumu":"cNDCG@3","test_NDCG_5_cumu":"cNDCG@5","test_disparity":"Unfairness tolerance"}
result_list=[]
positionBiasSeverities=[
    # "positionBiasSeverity_0",
    "positionBiasSeverity_1",
    # "positionBiasSeverity_2"
    ]
xMQfunctions=results_org.setScaleFunction(a=13600,b=.01,low=True)
xMSLRfunctions=results_org.setScaleFunction(a=0,b=.01,low=True)
xIsfunctions=[lambda x:x, lambda x:x]
yMQfunctions=results_org.setScaleFunction(a=202,b=1,low=False)
yMSLRfunctions=results_org.setScaleFunction(a=200,b=1,low=False)
# xIsfunctions=[lambda x: np.log(np.log(210-x)), lambda x:210-np.exp(np.exp(x))]
# yscaleFcn={"MQ2008":yMQfunctions,"MSLR10k":yIsfunctions}
# xscaleFcn={"MQ2008":xMQfunctions,"MSLR10k":xIsfunctions}
yscaleFcn={"MQ2008":xIsfunctions,"MSLR10k":yMSLRfunctions,"istella-s":xMSLRfunctions}
xscaleFcn={"MQ2008":xMQfunctions,"MSLR10k":xMSLRfunctions,"istella-s":xMSLRfunctions}
xticks={"MQ2008":[13800,15000,25000],"MSLR10k":[100,1000,5000],"istella-s":[10,30,150]}
yticks={"MQ2008":[185,190,195],"MSLR10k":[130,170,190],"istella-s":[110,140,190]}
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

        # result_validated["QPfairNDCG_500"]=result["fairness_strategy_QPfairNDCG"]["n_futureSession_500"]
        # result_validated["QPfairNDCG_500Hori"]=result["fairness_strategy_QPfairNDCGHorizontal"]["n_futureSession_500"]
        # result_validated["GradFair(Ours)_0"]=results_org.getGrandchildNode(result["fairness_strategy_GradFair"],"exploration_tradeoff_param_0.0")        
        # result_validated["GradFair(Ours)_0.1"]=results_org.getGrandchildNode(result["fairness_strategy_GradFair"],"exploration_tradeoff_param_0.1")
        # result_validated["GradFair(Ours)_0.5"]=results_org.getGrandchildNode(result["fairness_strategy_GradFair"],"exploration_tradeoff_param_0.5")
        # result_validated["GradFair(Ours)10"]=results_org.getGrandchildNode(result["fairness_strategy_GradFair"],"exploration_tradeoff_param_10")
        # result_validated["GradFair(Ours)50"]=results_org.getGrandchildNode(result["fairness_strategy_GradFair"],"exploration_tradeoff_param_50")
        # result_validated["QPFair"]=result["fairness_strategy_QPFair"]["n_futureSession_100"]
        # result_validated["QPFair-Horiz."]=result["fairness_strategy_QPFair-Horiz."]["n_futureSession_100"]
        # result_validated["QPFair (Ours)"]=results_org.getGrandchildNode(result["fairness_strategy_QPFair"]["n_futureSession_100"],"exploration_tradeoff_param_5")
        # result_validated["QPFair-Horiz."]=results_org.getGrandchildNode(result["fairness_strategy_QPFair-Horiz."]["n_futureSession_100"],"exploration_tradeoff_param_5")
        result_validated["FARA"]=results_org.getGrandchildNode(result["fairness_strategy_QPFair"]["n_futureSession_100"],"exploration_tradeoff_param_10")
        # result_validated["QPFair-Horiz."]=results_org.getGrandchildNode(result["fairness_strategy_QPFair-Horiz."]["n_futureSession_100"],"exploration_tradeoff_param_10")
        result_validated["FARA-w/o-Exp."]=results_org.getGrandchildNode(result["fairness_strategy_QPFair"]["n_futureSession_100"],"exploration_tradeoff_param_0.0")
        # result_validated["QPFair-Horiz.w/o Expl"]=results_org.getGrandchildNode(result["fairness_strategy_QPFair-Horiz."]["n_futureSession_100"],"exploration_tradeoff_param_0.0")
        result_validated=results_org.reorderDict(result_validated,config.desiredGradFair)
  
        result_validatedScatter={}
        # result_validatedScatter["TopK"]=result["fairness_strategy_Topk"]
        # result_validatedScatter["RandomK"]=result["fairness_strategy_Randomk"]
        # result_validatedScatter["FairK(Ours)"]=result["fairness_strategy_FairK"]
        # result_validatedScatter["ExploreK"]=result["fairness_strategy_ExploreK"]
        for ind,metrics in enumerate(metric_name):
#             fig, axs = plt.subplots(figsize=(6.4,2.4))
            fig, axs = plt.subplots()
            results_org.RequirementPlot(result_validated, metrics,ax=axs,step=step)
            
            for line in axs.lines:
#                 line.set_marker(None)
                line.set_linewidth(2.5)
            # results_org.TradeoffScatter(result_validatedScatter, metrics,ax=axs,step=step)
            axs.set_ylabel(metric_name_dict[metrics[1]])
            axs.set_xlabel(metric_name_dict[metrics[0]])
            # axs.set_title(data_name_cur)
            axs.set_xscale("function",functions=xscaleFcn[data_name_cur]) 
            axs.set_yscale("function",functions=yscaleFcn[data_name_cur]) 
            # axs.set_yscale("mylog2f")
            # axs[ind].set_xscale("log")
            # axs[0].set_yscale("symlog")
            # axs.legend(bbox_to_anchor=(1.1, 1.05)) 
            axs.legend()   
            axs.xaxis.set_major_formatter(mticker.ScalarFormatter())
            axs.xaxis.get_major_formatter().set_scientific(False)
            axs.xaxis.get_major_formatter().set_useOffset(False)
            axs.set_xticks(ticks=xticks[data_name_cur])
            axs.set_yticks(ticks=yticks[data_name_cur])
            # plt.setp(plt.gca().get_legend().get_texts(), fontsize='12')
            fig.savefig(os.path.join(OutputPath,"Realworld"+positionBiasSeverity+data_name_cur+metrics[1]+"abalation.pdf"), dpi=300, bbox_inches = 'tight', pad_inches = 0.05)
            plt.close(fig)