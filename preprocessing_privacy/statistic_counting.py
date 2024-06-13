import json
import pickle
import argparse
import random
data_name = ["FB15K"]


if __name__ == "__main__":
    for dataname in data_name:
        test_query_dir = "../input_files_privacy/" + dataname + "_test_private_public_queries.pkl"
        with open(test_query_dir, "rb") as fin:
            test_data_dict = pickle.load(fin)

        print()
        statistics = {}
        for query_type, query_answer in test_data_dict.items():
            statistics[query_type] = {}
            statistics[query_type]["train"] = 0
            statistics[query_type]["valid"] = 0
            statistics[query_type]["test_private"] = 0
            statistics[query_type]["test_public"] = 0
            for query, answer in query_answer.items():
                statistics[query_type]["train"] += len(answer["train_answers"])
                statistics[query_type]["valid"] += len(answer["valid_answers"])
                statistics[query_type]["test_private"] += len(answer["test_private_answers"])
                statistics[query_type]["test_public"] += len(answer["test_public_answers"]) - len(answer["valid_answers"])

            statistics[query_type]["train"] /= 0
            statistics[query_type]["valid"] /= 0
            statistics[query_type]["test_private"] /= 0
            statistics[query_type]["test_public"] /= 0
