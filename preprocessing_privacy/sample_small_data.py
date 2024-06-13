import json
import pickle
import argparse
import random
data_name = ["FB15K"]




if __name__ == "__main__":
    for dataname in data_name:
        test_query_dir = "../input_files_small_privacy/" + dataname + "_small_test_queries.pkl"
        with open(test_query_dir, "rb") as fin:
            test_data_dict = pickle.load(fin)

        test_data_dict_small = {}
        for key in test_data_dict.keys():
            test_data_dict_small[key] = {}
            for key2 in test_data_dict[key].keys():
                test_data_dict_small[key][key2] = {}
                test_data_dict_small[key][key2]["train_answers"] = test_data_dict[key][key2]["train_answers"]
                test_data_dict_small[key][key2]["valid_answers"] = test_data_dict[key][key2]["valid_answers"]
                test_data_dict_small[key][key2]["test_private_answers"] = test_data_dict[key][key2]["test_answers"]
                test_data_dict_small[key][key2]["test_public_answers"] = test_data_dict[key][key2]["test_answers"]

        with open("../input_files_small_privacy/" + dataname + "_test_private_public_queries.pkl", "wb") as fout:
            pickle.dump(test_data_dict_small, fout)




