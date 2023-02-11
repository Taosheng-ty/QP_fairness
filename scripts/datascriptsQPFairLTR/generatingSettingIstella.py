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
        setting_data={**setting_data, **dataset_dict[dataset]} 
        print(root_path,"x"*100)
        tools.iterate_settings(list_settings_data,setting_data,path=root_path) 


settings_base={
        "progressbar":"false",
        "rankListLength":5,
        "query_least_size":5,
        # "NumDocMaximum":20,
        # "relvance_strategy":"EstimatedAverage"
        }
# root_path="localOutput/Feb182022Data/"
positionBiasSeverity=[1]
# root_path="localOutput/QPFairLTRistella/"
root_path="localOutput/Jan252023QPFairLTRistella/"
desired_order_list=["relvance_strategy",'positionBiasSeverity',"dataset_name","fairness_strategy","n_futureSession","fairness_tradeoff_param","exploration_tradeoff_param","random_seed"]

#################### for post-processing
##write setting.json for 'FairCo', 'FairCo_multip.','FairCo_average'
datasets=["istella-s"]
dataset_dict={"istella-s":{"n_iteration":4*10**6,"queryMaximumLength":int(1e10)},\
        "MSLR-WEB10k":{"n_iteration":4*10**6,"queryMaximumLength":int(1e10)}}

list_settings={"relvance_strategy":["TrueAverage"],'positionBiasSeverity':positionBiasSeverity,"fairness_strategy":['PLFair'],"fairness_tradeoff_param":[0.0,0.01,0.05,0.1,0.2,0.5,0.8,0.9,1.0],\
              "exploration_tradeoff_param":[0.0], "random_seed":[0,1,2,3,4],"n_futureSession":[10000000]}
write_setting(datasets,list_settings,settings_base)
##write setting.json for Topk and Randomk
list_settings={"relvance_strategy":["TrueAverage"],'positionBiasSeverity':positionBiasSeverity,"fairness_strategy":['MMF'],"fairness_tradeoff_param":[0.0,0.01,0.05,0.1,0.2,0.5,0.8,0.9,1.0],\
              "exploration_tradeoff_param":[0.0], "random_seed":[0,1,2,3,4]}
write_setting(datasets,list_settings,settings_base)

list_settings={"relvance_strategy":["TrueAverage"],'positionBiasSeverity':positionBiasSeverity,"fairness_strategy":['FairCo'],"fairness_tradeoff_param":[0.0,0.0001,0.001,0.005,0.01,0.1,0.5,1,10,50,100,500,700,1000],\
              "exploration_tradeoff_param":[0.0], "random_seed":[0,1,2,3,4]}
write_setting(datasets,list_settings,settings_base)


##write setting.json for Topk and Randomk

list_settings={"relvance_strategy":["TrueAverage"],'positionBiasSeverity':positionBiasSeverity,"fairness_strategy":['Topk','Randomk',"FairK"],"random_seed":[0,1,2,3,4]}
write_setting(datasets,list_settings,settings_base)

##write setting.json for ILP and LP  only for MQ2008
settings_base_LP=dict(settings_base)
settings_base_LP["LogTimeEachStep"]="True"
list_settings={"relvance_strategy":["TrueAverage"],'positionBiasSeverity':positionBiasSeverity,"fairness_strategy":[ 'ILP'],"fairness_tradeoff_param":[1.0],\
               "exploration_tradeoff_param":[0.0],"random_seed":[0,1,2,3,4]}
write_setting(datasets,list_settings,settings_base_LP)
list_settings={"relvance_strategy":["TrueAverage"],'positionBiasSeverity':positionBiasSeverity,"fairness_strategy":[ 'LP'],"fairness_tradeoff_param":[1000],\
               "exploration_tradeoff_param":[0.0],"random_seed":[0],"n_futureSession":[100]}
# settings_base_LP=dict(settings_base)
# settings_base_LP["n_futureSession"]=200
write_setting(datasets,list_settings,settings_base_LP) 

##write setting.json for 'QPfair',"Hybrid","QPfairNDCG"

# list_settings={"relvance_strategy":["TrueAverage"],'positionBiasSeverity':positionBiasSeverity,"fairness_strategy":["QPFair","QPFair-Horiz."],"fairness_tradeoff_param":[0.0,0.01,0.05,0.1,0.2,0.5,0.8,0.85,0.9,0.92,0.95,0.98,1.0],\
#                "exploration_tradeoff_param":[0.0],"random_seed":[0,1,2,3,4],"n_futureSession":[2,5,10,20,50,100,200,500]}
# write_setting(datasets,list_settings,settings_base)

list_settings={"relvance_strategy":["TrueAverage"],'positionBiasSeverity':positionBiasSeverity,"fairness_strategy":["QPFair","QPFair-Horiz."],"fairness_tradeoff_param":[0.0,0.01,0.05,0.1,0.2,0.5,0.8,0.85,0.9,0.92,0.95,0.98,1.0],\
               "exploration_tradeoff_param":[0.0],"random_seed":[0,1,2,3,4],"n_futureSession":[100]}
write_setting(datasets,list_settings,settings_base)
# list_settings={"relvance_strategy":["TrueAverage"],'positionBiasSeverity':positionBiasSeverity,"fairness_strategy":["QPFair","QPFair-Horiz."],"fairness_tradeoff_param":[1.0],\
#                "exploration_tradeoff_param":[0.0],"random_seed":[0,1,2,3,4],"n_futureSession":[20,50,100,200,500]}
# write_setting(datasets,list_settings,settings_base)



#################### for in-processing

list_settings={"relvance_strategy":["EstimatedAverage"],'positionBiasSeverity':positionBiasSeverity,"fairness_strategy":['PLFair'],"fairness_tradeoff_param":[0.0,0.01,0.05,0.1,0.2,0.5,0.8,0.9,1.0],\
              "exploration_tradeoff_param":[0.0], "random_seed":[0,1,2,3,4],"n_futureSession":[int(dataset_dict["MSLR-WEB10k"]["n_iteration"]/10)]}
write_setting(datasets,list_settings,settings_base)
list_settings={"relvance_strategy":["EstimatedAverage"],'positionBiasSeverity':positionBiasSeverity,"fairness_strategy":['MMF'],"fairness_tradeoff_param":[0.0,0.01,0.05,0.1,0.2,0.5,0.8,0.9,1.0],\
              "exploration_tradeoff_param":[0.0], "random_seed":[0,1,2,3,4]}
write_setting(datasets,list_settings,settings_base)

# list_settings={"relvance_strategy":["EstimatedAverage"],'positionBiasSeverity':positionBiasSeverity,"fairness_strategy":[ 'GradFair'],"fairness_tradeoff_param":[0.0,0.0001,0.001,0.005,0.01,0.1,0.5,1,10,50,100,500,700,1000],\
#                "exploration_tradeoff_param":[0.0,0.1,0.5,1,5,10,20,50,100],"random_seed":[0,1,2,3,4]}
# write_setting(datasets,list_settings,settings_base)

list_settings={"relvance_strategy":["EstimatedAverage"],'positionBiasSeverity':positionBiasSeverity,"fairness_strategy":['FairCo'],"fairness_tradeoff_param":[0.0,0.00001,0.0001,0.0005,0.001,0.005,0.01,0.1,0.5,1,10,50,100,500,700,1000],\
               "exploration_tradeoff_param":[0.0],"random_seed":[0,1,2,3,4]}
write_setting(datasets,list_settings,settings_base)

# list_settings={"relvance_strategy":["EstimatedAverage"],'positionBiasSeverity':positionBiasSeverity,"fairness_strategy":["onlyFairness"],"fairness_tradeoff_param":[1],\
#                "exploration_tradeoff_param":[0.0,0.1,0.5,1,5,10,20,100,200,1000],"random_seed":[0,1,2,3,4]}
# write_setting(datasets,list_settings,settings_base)

##write setting.json for Topk and Randomk

list_settings={"relvance_strategy":["EstimatedAverage"],'positionBiasSeverity':positionBiasSeverity,"fairness_strategy":['Topk','Randomk',"FairK"],"random_seed":[0,1,2,3,4]}
write_setting(datasets,list_settings,settings_base)

##write setting.json for ILP and LP  only for MQ2008
# list_settings={"relvance_strategy":["EstimatedAverage"],'positionBiasSeverity':positionBiasSeverity,"fairness_strategy":[ 'ILP'],"fairness_tradeoff_param":[0.0,0.01,0.1,0.2,0.5,0.8,0.9,1.0],\
#                "exploration_tradeoff_param":[0.0],"random_seed":[0,1,2,3,4]}
# write_setting(datasets,list_settings,settings_base) 

# list_settings={"relvance_strategy":["EstimatedAverage"],'positionBiasSeverity':positionBiasSeverity,"fairness_strategy":[ 'LP'],"fairness_tradeoff_param":[0.0,0.01,0.05,0.1,0.2,0.5,1.0,2,10,50,80,100,500,1000],\
#                "exploration_tradeoff_param":[0.0],"random_seed":[0,1,2,3,4],"n_futureSession":[100]}
# # settings_base_LP=dict(settings_base)
# settings_base_LP["n_futureSession"]=200
# write_setting(datasets,list_settings,settings_base) 




#write setting.json for 'QPfair',"Hybrid","QPfairNDCG" in realworld setting


list_settings={"relvance_strategy":["EstimatedAverage"],'positionBiasSeverity':positionBiasSeverity,"fairness_strategy":["QPFair","QPFair-Horiz."],"fairness_tradeoff_param":[0.0,0.01,0.05,0.1,0.2,0.5,0.8,0.85,0.9,0.92,0.95,0.98,1.0],\
               "exploration_tradeoff_param":[0.0,10],"random_seed":[0,1,2,3,4],"n_futureSession":[100]}
write_setting(datasets,list_settings,settings_base)
# list_settings={"relvance_strategy":["EstimatedAverage"],'positionBiasSeverity':positionBiasSeverity,"fairness_strategy":["QPFair","QPFair-Horiz."],"fairness_tradeoff_param":[1.0],\
#                "exploration_tradeoff_param":[0.0,10],"random_seed":[0,1,2,3,4],"n_futureSession":[20,50,100,200,500]}
# write_setting(datasets,list_settings,settings_base)
