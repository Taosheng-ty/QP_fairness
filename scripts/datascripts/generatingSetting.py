import sys
import json
import os
scriptPath=os.path.dirname(os.path.abspath(__file__))
os.chdir(scriptPath+"/../..")
import BatchExpLaunch.tools as tools
def write_setting(datasets,list_settings,settings_base,shown_prob=None):
    """
    This function write settings.json which specify the parameters of main function.
    """
    
    for dataset in datasets:
        list_settings_data=dict(list_settings)
        list_settings_data["dataset_name"]=[dataset]
        list_settings_data = {k: list_settings_data[k] for k in desired_order_list if k in list_settings_data}
        setting_data=dict(settings_base)
        setting_data["n_iteration"]=int(iterations_dict[dataset]) 
        tools.iterate_settings(list_settings_data,setting_data,path=root_path) 

datasets=["Movie","News"]
iterations_dict={"Movie":6*10**3,"News":6*10**3}
settings_base={
        "progressbar":"false",
        "rankListLength":10,
        # "relvance_strategy":"EstimatedAverage"
        }
# root_path="localOutput/Feb182022Data/"
root_path="localOutput/Feb222022Data"
desired_order_list=["relvance_strategy",'positionBiasSeverity',"dataset_name","fairness_strategy","n_futureSession","fairness_tradeoff_param","random_seed"]
##write setting.json for 'FairCo', 'FairCo_multip.','FairCo_average'
list_settings={"relvance_strategy":["TrueAverage","EstimatedAverage"],'positionBiasSeverity':[0,1,2],"fairness_strategy":['FairCo', 'FairCo_average',"FairCo_maxnorm"],"fairness_tradeoff_param":[0.0,0.01,0.05,0.1,0.2,0.5,1,2,5,10,20,50,80,100,200,500,700,1000],\
               "random_seed":[0,1,2,3,4]}

write_setting(datasets,list_settings,settings_base)

##write setting.json for 'QPfair',"Hybrid","QPfairNDCG"

list_settings={"relvance_strategy":["TrueAverage","EstimatedAverage"],'positionBiasSeverity':[0,1,2],"fairness_strategy":["QPfairNDCG","QPfair"],"fairness_tradeoff_param":[0.0,0.01,0.05,0.1,0.2,0.5,0.8,0.85,0.9,0.92,0.95,0.98,1.0],\
               "random_seed":[0,1,2,3,4],"n_futureSession":[2,5,10,20,50,100,200]}
write_setting(datasets,list_settings,settings_base)

##write setting.json for Topk and Randomk

list_settings={"relvance_strategy":["TrueAverage","EstimatedAverage"],'positionBiasSeverity':[0,1,2],"fairness_strategy":['Topk','Randomk'],"random_seed":[0,1,2,3,4]}
write_setting(datasets,list_settings,settings_base)
