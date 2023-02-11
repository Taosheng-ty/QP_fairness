import sys
import os
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib
import numpy as np
plt.rcParams['pdf.fonttype']=42
font = {'size'   : 24}
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
# path_root="localOutput/QPFairLTRistella/relvance_strategy_EstimatedAverage"
path_root="localOutput/Apr30QPFairLTR/relvance_strategy_EstimatedAverage"
path_root="localOutput/July3QPFairLTR/relvance_strategy_EstimatedAverage"
# path_root="localOutput/Jan252023QPFairLTRistella/relvance_strategy_EstimatedAverage"
# path_root="localOutput/July3QPFairLTRMSLR/relvance_strategy_EstimatedAverage"
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

metric_name_dict={"test_NDCG_1_aver":"NDCG@1","test_NDCG_3_aver":"NDCG@3","test_NDCG_5_aver":"NDCG@5",\
    "test_NDCG_1_cumu":"cNDCG","test_NDCG_3_cumu":"cNDCG@3","test_NDCG_5_cumu":"cNDCG@5",\
                  "test_disparity":"Unfairness tolerance"}
result_list=[]
yMQfunctions=results_org.setScaleFunction(a=210,b=1,low=False)
yIsfunctions=results_org.setScaleFunction(a=210,b=1,low=False)

xMQfunctions=results_org.setScaleFunction(a=10,b=1,low=True)
xISfunctions=results_org.setScaleFunction(a=5,b=1,low=True)
# xMQfunctions=results_org.setScaleFunction(a=2.2*10**5,b=1,low=False)
eye=[lambda x:x, lambda x:x]
# xMQfunctions=results_org.setScaleFunction(a=-3,b=1,low=True)
# xIsfunctions=[lambda x: np.log(np.log(210-x)), lambda x:210-np.exp(np.exp(x))]
# yscaleFcn={"MQ2008":yMQfunctions,"MSLR10k":yIsfunctions}
# xscaleFcn={"MQ2008":xMQfunctions,"MSLR10k":xIsfunctions}
trO=lambda x:scale.SymmetricalLogTransform(base=10,linthresh=7,linscale=10).transform_non_affine(x-188)
intr=lambda x:188+scale.SymmetricalLogTransform(base=10,linthresh=7,linscale=10).transform_non_affine(x)
MQ2008Yaffine=[trO,intr]
trOX=lambda x:scale.SymmetricalLogTransform(base=10,linthresh=5000,linscale=3).transform_non_affine(x-20000)
intrX=lambda x:20000+scale.SymmetricalLogTransform(base=10,linthresh=5000,linscale=3).transform_non_affine(x)
MQ2008Xaffine=[trOX,intrX]
logscale=[lambda x: np.log(x-9000),lambda x: np.exp(x)+9000]
yscaleFcn={"MQ2008":MQ2008Yaffine,"MSLR10k":eye,"istella-s":eye}
xscaleFcn={"MQ2008":MQ2008Xaffine,"MSLR10k":xMQfunctions,"istella-s":xISfunctions}
xLimi={"MQ2008test_NDCG_1_cumu":[13100, 18000, 191, 201],"MQ2008test_NDCG_3_cumu":[13300, 26000, 185, 201],"MQ2008test_NDCG_5_cumu":[13300, 26000, 185, 201]}
ylimit={"MQ2008":[20,210]}
positionBiasSeverities=[
    # "positionBiasSeverity_0",
    "positionBiasSeverity_1",
    # "positionBiasSeverity_2"
    ]
xticks={"MQ2008":[10000,20000,300000],"MSLR10k":[100,1000,5000],"istella-s":[10,100,1000]}
yticks={"MQ2008":[150,190,200],"MSLR10k":[70,130,180],"istella-s":[50,100,200]}
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
#         result,result_mean=results_org.get_result_df(resultPath,groupby="iterations",rerun=True)
        result,result_mean=results_org.get_result_df(resultPath,groupby="iterations")
        result_validated["FairCo"]=result["fairness_strategy_FairCo"]
        # result_validated["FairCo_maxnorm"]=result["fairness_strategy_FairCo_maxnorm"]
        # result_validated["FairCo_multip."]=result["fairness_strategy_FairCo_multip."]
        # result_validated["LP_1"]=result["fairness_strategy_LP"]["n_futureSession_1"]
        if "fairness_strategy_LP" in result:
            result_validated["LP"]=result["fairness_strategy_LP"]["n_futureSession_100"]

            # result_validated["GradFair(Ours)"]=result["fairness_strategy_FairCo_average"]
            result_validated["ILP"]=result["fairness_strategy_ILP"]
        result_validated["MMF"]=result["fairness_strategy_MMF"]
        futureList=["n_futureSession_20000","n_futureSession_400000"]
        for future in futureList:
            if future in result["fairness_strategy_PLFair"]:
                result_validated["PLFair"]=result["fairness_strategy_PLFair"][future]
        result_validated["MMF"]=result["fairness_strategy_MMF"]
        for method in result_validated:
            result_validated[method]=results_org.getGrandchildNode(result_validated[method],"exploration_tradeoff_param_0.0")
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
        result_validated["FARA(Ours)"]=results_org.getGrandchildNode(result["fairness_strategy_QPFair"]["n_futureSession_100"],"exploration_tradeoff_param_10")
        result_validated["FARA-Horiz.(Ours)"]=results_org.getGrandchildNode(result["fairness_strategy_QPFair-Horiz."]["n_futureSession_100"],"exploration_tradeoff_param_10")
        # result_validated["QPFair (Ours)0"]=results_org.getGrandchildNode(result["fairness_strategy_QPFair"]["n_futureSession_100"],"exploration_tradeoff_param_0.0")
        # result_validated["QPFair-Horiz.0"]=results_org.getGrandchildNode(result["fairness_strategy_QPFair-Horiz."]["n_futureSession_100"],"exploration_tradeoff_param_0.0")
        result_validated=results_org.reorderDict(result_validated,config.desiredGradFair)
  
        result_validatedScatter={}
        result_validatedScatter["TopK"]=result["fairness_strategy_Topk"]
        result_validatedScatter["RandomK"]=result["fairness_strategy_Randomk"]
        # result_validatedScatter["FairK(Ours)"]=result["fairness_strategy_FairK"]
        # result_validatedScatter["ExploreK"]=result["fairness_strategy_ExploreK"]
        for ind,metrics in enumerate(metric_name):
            # fig, axs = plt.subplots()
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
                recPosition=[0.3, 0.4, 0.5, 0.4]
                axins = axs.inset_axes(recPosition)
                results_org.RequirementPlot(result_validated, metrics,\
                                            desiredColorDict=config.desiredGradFairColor,\
                                            desiredMarkerDict=config.desiredGradFairMarker,ax=axins,step=step)
                results_org.TradeoffScatter(result_validatedScatter, metrics,\
                                            desiredColorDict=config.desiredGradFairColor,ax=axins,step=step)
                x1, x2, y1, y2 = xLimi[data_name_cur+metrics[1]]
                axins.set_xlim(x1, x2)
                axins.set_ylim(y1, y2)
                axins.set_xscale("function",functions=xscaleFcn[data_name_cur]) 
                axs.indicate_inset_zoom(axins, edgecolor="black",alpha=1)
                axins.set_xticklabels([])
                axins.set_yticklabels([])
            # axs.set_title(data_name_cur)
            
            axs.set_xscale("function",functions=xscaleFcn[data_name_cur]) 
            axs.set_yscale("function",functions=yscaleFcn[data_name_cur]) 
            # axs.set_xscale("function",functions=xscaleFcn[data_name_cur]) 
            # axs.set_yscale("function",functions=yscaleFcn[data_name_cur]) 
            # if "MSLR" in data_name_cur:
            #     axs.set_xscale("log")
            legend,handles,labels=results_org.reorderLegend(config.desiredGradFair,axs,returnHandles=True)
            plt.setp(plt.gca().get_legend().get_texts(), fontsize='14')
            resultpath=os.path.join(OutputPath,positionBiasSeverity+data_name_cur+"Realworld")
            legend = axs.legend(handles, labels, loc=3,ncol=10, framealpha=1, frameon=True,bbox_to_anchor=(1.1, 1.05),columnspacing=0.5)
            results_org.export_legend(legend,resultpath+'legend.pdf')
            legend.remove()
            # plt.locator_params(axis='x', nbins=3)
            if data_name_cur in ylimit:
                axs.set_ylim(ylimit[data_name_cur])
            axs.set_xticks(ticks=xticks[data_name_cur])
            axs.set_yticks(ticks=yticks[data_name_cur])
            # plt.locator_params(axis='y', nbins=4) 
            # axs.legend(bbox_to_anchor=(1.1, 1.05)) 
            # axs.legend()   
            # results_org.reorderLegend(config.desiredGradFair,axs)
            # plt.setp(plt.gca().get_legend().get_texts(), fontsize='12')
            fig.savefig(os.path.join(OutputPath,"Realworld"+positionBiasSeverity+data_name_cur+metrics[1]+"tradeoffplot.pdf"), dpi=300, bbox_inches = 'tight', pad_inches = 0.05)
            plt.close(fig)