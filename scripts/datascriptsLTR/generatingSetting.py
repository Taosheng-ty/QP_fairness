import sys
import json
import os
scriptPath=os.path.dirname(os.path.abspath(__file__))
os.chdir(scriptPath+"/..")
import BatchExpLaunch.tools as tools
def write_setting(datasets,list_settings,settings_base,shown_prob):
    """
    This function write settings.json which specify the parameters of main function.
    """
    
    for dataset in datasets:
        list_settings_data=dict(list_settings)
        list_settings_data["dataset"]=[dataset]
        list_settings_data = {k: list_settings_data[k] for k in desired_order_list if k in list_settings_data}
        setting_data=dict(settings_base)
        setting_data["n_iteration"]=int(iterations_dict[dataset]/shown_prob) 
        tools.iterate_settings(list_settings_data,setting_data,path=root_path) 

datasets=["MQ2007","MQ2008","Webscope_C14_Set1","MSLR-WEB10k","MSLR-WEB30k"]
iterations_dict={"MSLR-WEB30k_beh_rm":3.8*10**6,"MSLR-WEB10k_beh_rm":1.2*10**6,\
                 "MSLR-WEB30k":3.8*10**6,"MSLR-WEB10k":1.2*10**6,\
                 "MQ2007":67*10**3,"MQ2008":15*10**3,"istella-s":10**6,\
                "Webscope_C14_Set1":687500,"MSLR-WEB30k_beh_rm1%":2*10**4,"NP2003":146*10**3,"NP2004":71*10**3,}
settings_base={
          "rankListLength":5,
        "query_least_size":5,
        "progressbar":"false"}
root_path="localOutput/Feb142022/"
shown_prob=0.05
desired_order_list=['positionBiasSeverity',"dataset","fairness_strategy","n_futureSession","fairness_tradeoff_param","random_seed"]
##write setting.json for 'FairCo', 'FairCo_multip.','FairCo_average'
list_settings={'positionBiasSeverity':[0,1,2],"fairness_strategy":['FairCo', 'FairCo_multip.','FairCo_average'],"fairness_tradeoff_param":[0.0,0.01,0.05,0.1,0.2,0.5,1,2,5,10,20,50,80,100,200,500,700,1000],\
               "random_seed":[0,1,2,3,4]}

write_setting(datasets,list_settings,settings_base,shown_prob)

##write setting.json for QPfair

list_settings={'positionBiasSeverity':[0,1,2],"fairness_strategy":['QPfair'],"fairness_tradeoff_param":[0.0,0.01,0.05,0.1,0.2,0.5,0.8,1.0],\
               "random_seed":[0,1,2,3,4],"n_futureSession":[2,5,10,20,50,100,200]}
write_setting(datasets,list_settings,settings_base,shown_prob)

##write setting.json for Topk and Randomk

list_settings={'positionBiasSeverity':[0,1,2],"fairness_strategy":['Topk','Randomk'],"random_seed":[0,1,2,3,4]}
write_setting(datasets,list_settings,settings_base,shown_prob)
