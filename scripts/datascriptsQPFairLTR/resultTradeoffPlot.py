import sys
import os
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib
import numpy as np
plt.rcParams['pdf.fonttype']=42
font = {'size'   : 24}
matplotlib.rc('font', **font)
# matplotlib.rcParams['lines.linewidth'] = 5
plt.rcParams['lines.linewidth'] = 3.0
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
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
path_root="localOutput/Apr252022LTR_small/relvance_strategy_TrueAverage"
path_root="localOutput/Apr262022LTR/relvance_strategy_TrueAverage"
path_root="localOutput/QPFairLTR/relvance_strategy_TrueAverage"
path_root="localOutput/QPFairLTRistella/relvance_strategy_TrueAverage"
path_root="localOutput/Apr30QPFairLTR/relvance_strategy_TrueAverage"
path_root="localOutput/July3QPFairLTR/relvance_strategy_TrueAverage"
# path_root="localOutput/Jan252023QPFairLTRistella/relvance_strategy_TrueAverage"
# path_root="localOutput/July3QPFairLTRMSLR/relvance_strategy_TrueAverage"
step=19  
data_rename={            
            # "Movie":"Movie",\
            # "News":"News",\
            # "MSLR-WEB30k":"MSLR-WEB30k",\
            "MSLR-WEB10k":"MSLR10k",\
            # "Webscope_C14_Set1":"Webscope_C14_Set1",\
            "istella-s":"istella-s",
            "MQ2008":"MQ2008",
            # "MQ2007":"MQ2007",
            # "istella-s":"ist",
}
metric_name=[["test_disparity",'test_NDCG_1_aver'],["test_disparity",'test_NDCG_3_aver'],\
    ["test_disparity",'test_NDCG_5_aver'],["test_disparity",'test_NDCG_1_cumu'],["test_disparity",'test_NDCG_3_cumu'],\
    ["test_disparity",'test_NDCG_5_cumu'],]
metric_name=[["test_disparity",'test_NDCG_1_cumu'],["test_disparity",'test_NDCG_3_cumu'],\
    ["test_disparity",'test_NDCG_5_cumu']]
# metric_name=[["disparity",'NDCG_3_aver'],["disparity",'NDCG_5_aver']]
xticks={"MQ2008":[9500,15000,100000],"MSLR10k":[0,1000,5000],"istella-s":[0,100,1000]}
yticks={"MQ2008":[100,180,200]}
ylimit={"MQ2008":[80,202]}
metric_name_dict={"test_NDCG_1_aver":"NDCG@1","test_NDCG_3_aver":"NDCG@3","test_NDCG_5_aver":"NDCG@5",\
    "test_NDCG_1_cumu":"cNDCG","test_NDCG_3_cumu":"cNDCG@3","test_NDCG_5_cumu":"cNDCG@5",\
                  "test_disparity":"Unfairness tolerance"}
result_list=[]
# yMQfunctions=results_org.setScaleFunction(a=210,b=1,low=False)
# yIsfunctions=results_org.setScaleFunction(a=210,b=1,low=False)

xMQfunctions=results_org.setScaleFunction(a=1000,b=1,low=True)
xMSLRfunctions=results_org.setScaleFunction(a=-3,b=1,low=True)
xIsfunctions=[lambda x:x, lambda x:x]
logscale=[lambda x: np.log(x-9000),lambda x: np.exp(x)+9000]
# yscaleFcn={"MQ2008":yMQfunctions,"MSLR10k":yIsfunctions}
# xscaleFcn={"MQ2008":xMQfunctions,"MSLR10k":xIsfunctions}
trO=lambda x:scale.SymmetricalLogTransform(base=10,linthresh=20,linscale=2).transform_non_affine(x-201)
intr=lambda x:201+scale.SymmetricalLogTransform(base=10,linthresh=20,linscale=2).transform_non_affine(x)
MQ2008Yaffine=[trO,intr]
trOX=lambda x:scale.SymmetricalLogTransform(base=10,linthresh=10000,linscale=3).transform_non_affine(x-9800)
intrX=lambda x:9800+scale.SymmetricalLogTransform(base=10,linthresh=10000,linscale=3).transform_non_affine(x)
MQ2008Xaffine=[trOX,intrX]
yscaleFcn={"MQ2008":MQ2008Yaffine,"MSLR10k":xIsfunctions,"istella-s":xIsfunctions}
xscaleFcn={"MQ2008":logscale,"MSLR10k":xMSLRfunctions,"istella-s":xMSLRfunctions}
# result_path=os.path.join(path_root,"result")
# for datasets,data_name_cur in data_rename.items():
#     fig, axs = plt.subplots(len(metric_name),figsize=(5,3*len(metric_name)), sharex=True)
#     result_validated={}
#     datasets="dataset_"+datasets
#     path=os.path.join(path_root,datasets)

# desiredGradFair=["Topk","ExploreK","FairCo","ILP","LP","FairK(Ours)","GradFair(Ours)"]

positionBiasSeverities=[
    # "positionBiasSeverity_0",
    "positionBiasSeverity_1",
    # "positionBiasSeverity_2"
    ]
result_list=[]
for positionBiasSeverity in positionBiasSeverities:
    OutputPath=os.path.join(path_root,"result")
    os.makedirs(OutputPath, exist_ok=True)
    for datasets,data_name_cur in data_rename.items():
        
        result_validated={}
        datasets="dataset_name_"+datasets
        resultPath=os.path.join(path_root,positionBiasSeverity,datasets)

        if not os.path.isdir(resultPath):
            # print(path)       
            continue
        # result,result_mean=results_org.get_result_df(resultPath,groupby="iterations",rerun=True)
        result,result_mean=results_org.get_result_df(resultPath,groupby="iterations")
        result_validated["FairCo"]=result["fairness_strategy_FairCo"]
        # result_validated["FairCo_maxnorm"]=result["fairness_strategy_FairCo_maxnorm"]
        # result_validated["FairCo_multip."]=result["fairness_strategy_FairCo_multip."]
        # result_validated["LP_1"]=result["fairness_strategy_LP"]["n_futureSession_1"]
        result_validated["FARA(Ours)"]=result["fairness_strategy_QPFair"]["n_futureSession_100"]
        result_validated["FARA-Horiz.(Ours)"]=result["fairness_strategy_QPFair-Horiz."]["n_futureSession_100"]
        if "fairness_strategy_LP" in result:
            result_validated["LP"]=result["fairness_strategy_LP"]["n_futureSession_100000"]
            result_validated["ILP"]=result["fairness_strategy_ILP"]
        result_validated["PLFair"]=result["fairness_strategy_PLFair"]["n_futureSession_10000000"]
        result_validated["MMF"]=result["fairness_strategy_MMF"]
        for method in result_validated:
            result_validated[method]=results_org.getGrandchildNode(result_validated[method],"exploration_tradeoff_param_0.0")
        result_validated=results_org.reorderDict(result_validated,config.desiredGradFair)
        result_validatedScatter={}
        result_validatedScatter["TopK"]=result["fairness_strategy_Topk"]
        result_validatedScatter["RandomK"]=result["fairness_strategy_Randomk"]
        # result_validatedScatter["FairK(Ours)"]=result["fairness_strategy_FairK"]
#         result_validatedScatter["RandomK"]=result["fairness_strategy_Randomk"]
        for ind,metrics in enumerate(metric_name):
            fig, axs = plt.subplots(figsize=(10,6))
            results_org.RequirementPlot(result_validated, metrics,\
                                        desiredColorDict=config.desiredGradFairColor,\
                                            desiredMarkerDict=config.desiredGradFairMarker,ax=axs,step=step)
            for line in axs.lines:
#                 line.set_marker(None)
                line.set_linewidth(2.5)
            results_org.TradeoffScatter(result_validatedScatter, metrics,\
                                        desiredColorDict=config.desiredGradFairColor,ax=axs,step=step)
            axs.set_ylabel(metric_name_dict[metrics[1]])
            axs.set_xlabel(metric_name_dict[metrics[0]])

            if "M1Q" in data_name_cur:
                recPosition=[0.45, 0.60, 0.45, 0.380]
                axins = axs.inset_axes(recPosition)
                results_org.RequirementPlot(result_validated, metrics,\
                                            desiredColorDict=config.desiredGradFairColor,\
                                            desiredMarkerDict=config.desiredGradFairMarker,ax=axins,step=step)
                results_org.TradeoffScatter(result_validatedScatter, metrics,\
                                            desiredColorDict=config.desiredGradFairColor,ax=axins,step=step)
                x1, x2, y1, y2 = 8000, 11000, 150, 202
                axins.set_xlim(x1, x2)
                axins.set_ylim(y1, y2)
                axs.indicate_inset_zoom(axins, edgecolor="black",alpha=1 )
                axins.set_xticklabels([])
                axins.set_yticklabels([])
            # axs.set_title(data_name_cur)
            axs.set_xscale("function",functions=xscaleFcn[data_name_cur]) 
            axs.set_yscale("function",functions=yscaleFcn[data_name_cur]) 
            # axs.set_xscale("function",functions=xscaleFcn[data_name_cur]) 
            # axs.set_yscale("function",functions=yscaleFcn[data_name_cur]) 
            # if "MSLR" in data_name_cur:
            #     axs.set_xscale("log")
            # if "MQ" in data_name_cur:
            #     x1, x2, y1, y2 = 9000, 17000, 180, 202
            #     axs.set_xlim(x1, x2)
            #     axs.set_ylim(y1, y2)
            
            legend,handles,labels=results_org.reorderLegend(config.desiredGradFair,axs,returnHandles=True)
            plt.setp(plt.gca().get_legend().get_texts(), fontsize='12')
            resultpath=os.path.join(OutputPath,positionBiasSeverity+data_name_cur)
            legend = axs.legend(handles, labels, loc=3,ncol=10, framealpha=1, frameon=True,bbox_to_anchor=(1.1, 1.05),columnspacing=0.5)
            results_org.export_legend(legend,resultpath+'legend.pdf')
            legend.remove()
            axs.set_xticks(ticks=xticks[data_name_cur])
            if data_name_cur in ylimit:
                axs.set_ylim(ylimit[data_name_cur])
            if data_name_cur in yticks:
                axs.set_yticks(ticks=yticks[data_name_cur])
            plt.locator_params(axis='y', nbins=4) 
            # plt.locator_params(axis='y', nbins=4) 
            # axs.legend(bbox_to_anchor=(1.1, 1.05)) 
            # axs.legend()   
            # results_org.reorderLegend(config.desiredGradFair,axs)
            # plt.setp(plt.gca().get_legend().get_texts(), fontsize='12')
            fig.savefig(os.path.join(OutputPath,positionBiasSeverity+data_name_cur+metrics[1]+"tradeoffplot.pdf"), dpi=300, bbox_inches = 'tight', pad_inches = 0.05)
            plt.close(fig)