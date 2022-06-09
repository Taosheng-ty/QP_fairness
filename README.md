# QP_fairness
## To run GradFair
please first run the following to generate experiment settings,

            python scripts/datascriptsGradFairLTR/generatingSetting.py

With the setting.json generated from above cmd, you can run the following to run a batch of scripts.

            slurm_python --CODE_PATH=.  --Cmd_file=main.py --JSON_PATH=localOutput/Mar292022Data20Docs/ --python_ver=QPfairness --jobs_limit=20  --secs_each_sub=0.1 --json2args --plain_script   --white_list=ty_1+d_0+Mov+QP+expl --only_unfinished
