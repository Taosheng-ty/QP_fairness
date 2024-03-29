import sys
import os
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib
import numpy as np
plt.rcParams['pdf.fonttype']=42
font = {'size'   : 15}

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
path_root="localOutput/Apr262022LTR/relvance_strategy_EstimatedAverage"
# path_root="localOutput/Apr30QPFairLTR/relvance_strategy_EstimatedAverage"
from scipy.interpolate import make_interp_spline, BSpline
from scipy.signal import savgol_filter
import scipy
# import statsmodels.api as sm
def smooth(x,y, box_pts=2):
    if x.shape[0]<=1:
        return x,y
    x=x+np.arange(y.shape[0])/10000000
    xnew = np.linspace(x.min(), x.max(), 300) 
    spl = make_interp_spline(x, y, k=3)  # type: BSpline
    power_smooth = spl(xnew)
    return xnew,power_smooth

# def smooth(x,y, box_pts=2):
#     if x.shape[0]<=4:
#         return x,y
#     N=y.shape[0]
#     w = scipy.fft.rfft(y)
#     f = scipy.fft.rfftfreq(N, x[1]-x[0])
#     spectrum = w**2

#     cutoff_idx = spectrum < (spectrum.max()/5)
#     w2 = w.copy()
#     w2[cutoff_idx] = 0

#     y2 = scipy.fft.irfft(w2)
#     # print(x.shape,y2.shape,y.shape)
#     return x,y2
smooth=None
# def smooth(x,y, box_pts=3):
#     box = np.ones(box_pts)/box_pts
#     y_smooth = np.convolve(y, box, mode='valid')
#     return x[2:],y_smooth
# def smooth(x,y, box_pts=3):
#     if x.shape[0]<=2:
#         return x,y
#     poly = np.polyfit(x,y,2)
#     poly_y = np.poly1d(poly)(x)
#     return x,poly_y
step=19  
data_rename={            
            # "Movie":"Movie",\
            # "News":"News",\
            # "MSLR-WEB30k":"MSLR-WEB30k",\
            # "MSLR-WEB10k":"MSLR-WEB10k",\
            # "Webscope_C14_Set1":"Webscope_C14_Set1",\
            "istella-s":"ist",
            "MQ2008":"MQ2008",
            # "MQ2007":"MQ2007",
}
metric_name=[["test_disparity",'test_NDCG_1_aver'],["test_disparity",'test_NDCG_3_aver'],\
    ["test_disparity",'test_NDCG_5_aver'],["test_disparity",'test_NDCG_1_cumu'],["test_disparity",'test_NDCG_3_cumu'],\
    ["test_disparity",'test_NDCG_5_cumu'],]
# metric_name=[["test_disparity",'test_NDCG_5_cumu']]

metric_name_dict={"test_NDCG_1_aver":"NDCG@1","test_NDCG_3_aver":"NDCG@3","test_NDCG_5_aver":"NDCG@5",\
    "test_NDCG_1_cumu":"cNDCG@1","test_NDCG_3_cumu":"cNDCG@3","test_NDCG_5_cumu":"cNDCG@5","test_disparity":"Unfairness tolerance"}
result_list=[]
positionBiasSeverities=[
    # "positionBiasSeverity_0",
    "positionBiasSeverity_1",
    # "positionBiasSeverity_2"
    ]
yMQfunctions=results_org.setScaleFunction(a=201,b=1,low=False)
yIsfunctions=results_org.setScaleFunction(a=210,b=1,low=False)
yscaleFcn={"MQ2008":yMQfunctions,"ist":yIsfunctions}
xMQfunctions=results_org.setScaleFunction(a=10,b=1,low=True)
xIsfunctions=[lambda x:x, lambda x:x]
xscaleFcn={"MQ2008":xMQfunctions,"ist":xMQfunctions}
xticks={"MQ2008":[3000,20000,50000],"ist":[50,1000,5000]}
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
        result_validated["FairCo"]=result["fairness_strategy_FairCo"]
        # result_validated["FairCo_maxnorm"]=result["fairness_strategy_FairCo_maxnorm"]
        # result_validated["FairCo_multip."]=result["fairness_strategy_FairCo_multip."]
        # result_validated["LP_1"]=result["fairness_strategy_LP"]["n_futureSession_1"]
        if "fairness_strategy_LP" in result:
            result_validated["LP"]=result["fairness_strategy_LP"]["n_futureSession_100"]

            # result_validated["GradFair(Ours)"]=result["fairness_strategy_FairCo_average"]
            result_validated["ILP"]=result["fairness_strategy_ILP"]
        result_validated["MMF"]=result["fairness_strategy_MMF"]
        futureList=["n_futureSession_10000","n_futureSession_1000000"]
        for future in futureList:
            if future in result["fairness_strategy_PLFair"]:
                result_validated["PLFair"]=result["fairness_strategy_PLFair"][future]
        for method in result_validated:
            result_validated[method]=results_org.getGrandchildNode(result_validated[method],"exploration_tradeoff_param_0.0")
        # result_validated["QPfairNDCG_500"]=result["fairness_strategy_QPfairNDCG"]["n_futureSession_500"]
        # result_validated["QPfairNDCG_500Hori"]=result["fairness_strategy_QPfairNDCGHorizontal"]["n_futureSession_500"]
        # result_validated["GradFair(Ours)_0"]=results_org.getGrandchildNode(result["fairness_strategy_GradFair"],"exploration_tradeoff_param_0.0")        
        # result_validated["GradFair(Ours)_0.1"]=results_org.getGrandchildNode(result["fairness_strategy_GradFair"],"exploration_tradeoff_param_0.1")
        # result_validated["GradFair(Ours)_0.5"]=results_org.getGrandchildNode(result["fairness_strategy_GradFair"],"exploration_tradeoff_param_0.5")
        # result_validated["GradFair(Ours)"]=results_org.getGrandchildNode(result["fairness_strategy_GradFair"],"exploration_tradeoff_param_100")
        result_validated["MCFair(Ours)"]=results_org.getGrandchildNode(result["fairness_strategy_GradFair"],"exploration_tradeoff_param_100")

        result_validated=results_org.reorderDict(result_validated,config.desiredGradFair)
        # result_validated["GradFair(Ours)_1"]=results_org.getGrandchildNode(result["fairness_strategy_GradFair"],"exploration_tradeoff_param_1")
        # result_validated["GradFair(Ours)_5"]=results_org.getGrandchildNode(result["fairness_strategy_GradFair"],"exploration_tradeoff_param_5")
        # result_validated["GradFair(Ours)_0.5"]=results_org.getGrandchildNode(result["fairness_strategy_GradFair"],"exploration_tradeoff_param_0.5")        
        # result_validated["GradFair(Ours)_0.0"]=results_org.getGrandchildNode(result["fairness_strategy_GradFair"],"exploration_tradeoff_param_0.0")        

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
        result_validatedScatter={}
        result_validatedScatter["TopK"]=result["fairness_strategy_Topk"]
        # result_validatedScatter["RandomK"]=result["fairness_strategy_Randomk"]
        result_validatedScatter["FairK(Ours)"]=result["fairness_strategy_FairK"]
        result_validatedScatter["ExploreK"]=result["fairness_strategy_ExploreK"]
        for ind,metrics in enumerate(metric_name):
            fig, axs = plt.subplots()
            results_org.RequirementPlot(result_validated, metrics,\
                                        desiredColorDict=config.desiredGradFairColor,desiredMarkerDict=config.desiredGradFairMarker,ax=axs,step=step,smoooth_fn=smooth)
            for line in axs.lines:
#                 line.set_marker(None)
                line.set_linewidth(1.5)
            results_org.TradeoffScatter(result_validatedScatter, metrics,\
                                        desiredColorDict=config.desiredGradFairColor,ax=axs,step=step)
            axs.set_ylabel(metric_name_dict[metrics[1]])
            axs.set_xlabel(metric_name_dict[metrics[0]])
            # axs.set_title(data_name_cur)
            axs.set_xscale("function",functions=xscaleFcn[data_name_cur]) 
            axs.set_yscale("function",functions=yscaleFcn[data_name_cur]) 
            # plt.locator_params(axis='x', nbins=2)
            axs.set_xticks(ticks=xticks[data_name_cur])
            plt.locator_params(axis='y', nbins=3)
            # axs[ind].set_xscale("log")
            # axs[0].set_yscale("symlog")
            # axs.legend(bbox_to_anchor=(1.1, 1.05)) 
            # axs.legend()   
            legend,handles,labels=results_org.reorderLegend(config.desiredGradFair,axs,returnHandles=True)
            plt.setp(plt.gca().get_legend().get_texts(), fontsize='15')
            resultpath=os.path.join(OutputPath,positionBiasSeverity+data_name_cur)
            legend = axs.legend(handles, labels, loc=3,ncol=9, framealpha=1, frameon=True,bbox_to_anchor=(1.1, 1.05),columnspacing=0.5)
            results_org.export_legend(legend,resultpath+'legend.pdf')
            legend.remove()
            # axs.legend()
#             plt.locator_params(axis='both', nbins=4)
            fig.savefig(os.path.join(OutputPath,"RealworldRequirement"+positionBiasSeverity+data_name_cur+metrics[1]+"tradeoffplot.pdf"), dpi=600, bbox_inches = 'tight', pad_inches = 0.05)
            plt.close(fig)