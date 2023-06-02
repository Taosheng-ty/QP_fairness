## Create the env.
    conda env create -f environment.yml
then activate the env 

    conda activate FARA
## Specify the data directory 
Please download datasets and specify their directory in LTRlocal_dataset_info.txt.

## Then you can run our experiemtns with the following 
    python  ./main.py   --progressbar=false --rankListLength=5 --query_least_size=5 --n_iteration=4000000 --queryMaximumLength=10000000000 --relvance_strategy=TrueAverage --positionBiasSeverity=1 --dataset_name=istella-s --fairness_strategy=FARA --n_futureSession=100 --fairness_tradeoff_param=0.2 --exploration_tradeoff_param=0.0 --random_seed=4 --log_dir=localOutput/

You can choose different datasets, relevance_strategy(online(EstimatedAverage) or post-processing(TrueAverage)) , fairness_strategies.













