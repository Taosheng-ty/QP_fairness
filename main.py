import numpy as np
import utils.dataset as dataset
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str,
                        help="Path to result logs")
    parser.add_argument("--fold_id", type=int,
                        help="Fold number to select, modulo operator is applied to stay in range.",
                        default=1)
    parser.add_argument("--dataset", type=str,
                        default="Webscope_C14_Set1",
                        help="Name of dataset to sample from.")
    parser.add_argument("--dataset_info_path", type=str,
                        default="local_dataset_info.txt",
                        help="Path to dataset info file.")

    args = parser.parse_args()
    # load the data
    data = dataset.get_data(args.dataset,
                  args.dataset_info_path,
                  args.fold_id)
    # 