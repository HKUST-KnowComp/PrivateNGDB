import argparse
from gqe import GQE
from q2b import Q2B
from q2p import Q2P

import torch
from dataloader import TrainDataset, ValidDataset, TestDataset, SingledirectionalOneShotIterator, separate_query_dict
from dataloader import baseline_abstraction, abstraction
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np
from datetime import datetime
from tensorboardX import SummaryWriter
import gc
import pickle
from torch.optim.lr_scheduler import LambdaLR
import json
import networkx as nx
import time
import random
from util import pad_tensor


def log_aggregation(list_of_logs):
    all_log = {}

    for __log in list_of_logs:
        # Sometimes the number of answers are 0, so we need to remove all the keys with 0 values
        # The average is taken over all queries, instead of over all answers, as is done following previous work.
        ignore_exd = False
        ignore_ent = False
        ignore_inf = False

        if "exd_num_answers" in __log and __log["exd_num_answers"] == 0:
            ignore_exd = True
        if "ent_num_answers" in __log and __log["ent_num_answers"] == 0:
            ignore_ent = True
        if "inf_num_answers" in __log and __log["inf_num_answers"] == 0:
            ignore_inf = True

        for __key, __value in __log.items():
            if "num_answers" in __key:
                continue

            else:
                if ignore_ent and "ent_" in __key:
                    continue
                if ignore_exd and "exd_" in __key:
                    continue
                if ignore_inf and "inf_" in __key:
                    continue

                if __key in all_log:
                    all_log[__key].append(__value)
                else:
                    all_log[__key] = [__value]

    average_log = {_key: np.mean(_value) for _key, _value in all_log.items()}

    return average_log


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='The training and evaluation script for the models')

    parser.add_argument('--query_data_dir', default="sampled_data_small", help="The path to the sampled queries.")
    parser.add_argument('--kg_data_dir', default="KG_data/", help="The path the original kg data")

    parser.add_argument("--train_query_dir", required=True)
    parser.add_argument("--valid_query_dir", required=True)
    parser.add_argument("--test_query_dir", required=True)
    parser.add_argument("--test_private_query_dir", required=True)
    parser.add_argument("--train_private_query_dir")

    parser.add_argument('--log_steps', default=50000, type=int, help='train log every xx steps')
    parser.add_argument('-dn', '--data_name', type=str, required=True)
    parser.add_argument('-b', '--batch_size', default=64, type=int)

    parser.add_argument('--max_train_step', default=370000, type=int)

    parser.add_argument('-d', '--entity_space_dim', default=400, type=int)
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float)
    parser.add_argument('-wc', '--weight_decay', default=0.0000, type=float)

    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--dropout_rate", default=0.1, type=float)
    parser.add_argument('-ls', "--label_smoothing", default=0.0, type=float)

    parser.add_argument("--warm_up_steps", default=1000, type=int)

    parser.add_argument("-m", "--model", required=True)

    parser.add_argument("--small", action="store_true")

    parser.add_argument("--timing", action="store_true")

    parser.add_argument("-ga", "--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--num_private_steps", type=int, default=50, help="Number of private steps")

    parser.add_argument("--mi", action="store_true")

    parser.add_argument("--cl", action="store_true")

    parser.add_argument("--privacy_multiplier", type=float, default=1, help="Private tradeoff")

    args = parser.parse_args()

    data_name = args.data_name
    if "json" in args.train_query_dir:
        with open(args.train_query_dir, "r") as fin:
            train_data_dict = json.load(fin)
    else:
        with open(args.train_query_dir, "rb") as fin:
            train_data_dict = pickle.load(fin)

    if "json" in args.valid_query_dir:
        with open(args.valid_query_dir, "r") as fin:
            valid_data_dict = json.load(fin)
    else:
        with open(args.valid_query_dir, "rb") as fin:
            valid_data_dict = pickle.load(fin)

    if "json" in args.test_query_dir:
        with open(args.test_query_dir, "r") as fin:
            test_data_dict = json.load(fin)
    else:
        with open(args.test_query_dir, "rb") as fin:
            test_data_dict = pickle.load(fin)

    if "json" in args.test_private_query_dir:
        with open(args.test_private_query_dir, "r") as fin:
            private_data_1p_dict = json.load(fin)
    else:
        with open(args.test_private_query_dir, "rb") as fin:
            private_data_1p_dict = pickle.load(fin)

    data_dir = args.data_name

    if args.small:
        print("Load Train Graph " + data_dir)
        train_path = "../preprocessing_privacy/" + data_dir + "_small_train_with_units.pkl"
        # train_graph = nx.read_gpickle(train_path)
        with open(train_path, "rb") as fin:
            train_graph = pickle.load(fin)

        print("Load Test Graph " + data_dir)
        test_path = "../preprocessing_privacy/" + data_dir + "_small_test_with_units.pkl"
        # test_graph = nx.read_gpickle(test_path)
        with open(test_path, "rb") as fin:
            test_graph = pickle.load(fin)

    else:
        print("Load Train Graph " + data_dir)
        train_path = "../preprocessing_privacy/" + data_dir + "_train_with_units.pkl"
        # train_graph = nx.read_gpickle(train_path)
        with open(train_path, "rb") as fin:
            train_graph = pickle.load(fin)

        print("Load Test Graph " + data_dir)
        test_path = "../preprocessing_privacy/" + data_dir + "_test_with_units.pkl"
        # test_graph = nx.read_gpickle(test_path)
        with open(test_path, "rb") as fin:
            test_graph = pickle.load(fin)

    entity_counter = 0
    value_counter = 0
    all_values = []

    for u in test_graph.nodes():
        if isinstance(u, tuple):
            value_counter += 1
            all_values.append(u)
        elif isinstance(u, str):
            entity_counter += 1

    value_vocab = dict(zip(all_values, range(0, len(all_values))))

    relation_edges_list = []
    attribute_edges_list = []
    reverse_attribute_edges_list = []
    numerical_edges_list = []
    for u, v, a in test_graph.edges(data=True):
        if isinstance(u, tuple) and isinstance(v, tuple):
            for key, value in a.items():
                numerical_edges_list.append(key)
        elif isinstance(u, tuple):
            for key, value in a.items():
                reverse_attribute_edges_list.append(key)
        elif isinstance(v, tuple):
            for key, value in a.items():
                attribute_edges_list.append(key)
        elif isinstance(u, str) and isinstance(v, str):
            for key, value in a.items():
                relation_edges_list.append(key)

    relation_edges_list = list(set(relation_edges_list))
    attribute_edges_list = list(set(attribute_edges_list))
    reverse_attribute_edges_list = list(set(reverse_attribute_edges_list))
    numerical_edges_list = list(set(numerical_edges_list))

    nentity = entity_counter
    nvalue = value_counter
    # TODO: QI HU
    if args.small:
        nrelation = max(relation_edges_list) + 10
        nattribute = max(attribute_edges_list) + 10
        nnumerical_proj = 10
    else:
        nrelation = len(relation_edges_list)
        nattribute = len(attribute_edges_list)
        nnumerical_proj = len(numerical_edges_list)

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = '../logs/gradient_tape/' + current_time + "_" + args.model + "_" + data_name + '_evaluations'
    test_log_dir = '../logs/gradient_tape/' + current_time + "_" + args.model + "_" + data_name + '_evaluations'

    if args.small:
        train_log_dir = train_log_dir + "_small"
        test_log_dir = test_log_dir + "_small"

    train_log_dir = train_log_dir + "/train"
    test_log_dir = test_log_dir + "/test"

    # read private 1p queries
    private_1p_path = "../input_files_privacy/" + data_dir + "_private_1p_queries.json"
    with open(private_1p_path, "r") as file_handle:
        private_1p_read = json.load(file_handle)

    private_1p = []

    for u, v, attr in private_1p_read:
        if not isinstance(u, str):
            u = tuple(u)
        if not isinstance(v, str):
            v = tuple(v)
        private_1p.append((u, v, attr))

    if args.cl:
        if "json" in args.train_private_query_dir:
            with open(args.train_private_query_dir, "r") as fin:
                private_train_query = json.load(fin)
        else:
            with open(args.train_private_query_dir, "rb") as fin:
                private_train_query = pickle.load(fin)

    train_summary_writer = SummaryWriter(train_log_dir)
    test_summary_writer = SummaryWriter(test_log_dir)

    batch_size = args.batch_size

    # Test train iterators

    if args.mi:
        extended_private_query_iterators = {}
        private_ap = private_1p[0: len(private_1p) // 2]
        query_answer_dict = {}
        for u, v, attr in private_ap:
            query_answer_dict['(ap,({}),(e,({})))'.format(list(attr.keys())[0], u)] = {}
            query_answer_dict['(ap,({}),(e,({})))'.format(list(attr.keys())[0], u)]["train_answers"] = [list(v)]
        new_iterator = SingledirectionalOneShotIterator(DataLoader(
            TrainDataset(nentity, nrelation, query_answer_dict,
                         baseline=True, nattribute=nattribute, value_vocab=value_vocab),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=TrainDataset.collate_fn
        ))
        extended_private_query_iterators['(ap,(e))'] = new_iterator

        query_answer_dict = {}
        private_ap = private_1p[len(private_1p) // 2:]
        for u, v, attr in private_ap:
            query_answer_dict['(rap,({}),(nv,({},{})))'.format(list(attr.keys())[0], u[1], u[0])] = {}
            query_answer_dict['(rap,({}),(nv,({},{})))'.format(list(attr.keys())[0], u[1], u[0])]["train_answers"] = [v]
        new_iterator = SingledirectionalOneShotIterator(DataLoader(
            TrainDataset(nentity, nrelation, query_answer_dict,
                         baseline=True, nattribute=nattribute, value_vocab=value_vocab),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=TrainDataset.collate_fn
        ))
        extended_private_query_iterators['(rap,(nv))'] = new_iterator
    if args.cl:
        if "json" in args.train_private_query_dir:
            with open(args.train_private_query_dir, "r") as fin:
                train_private_data_dict = json.load(fin)
        else:
            with open(args.train_query_dir, "rb") as fin:
                train_private_data_dict = pickle.load(fin)

        extended_private_train_data_dict = {}
        extended_private_train_query_types = []
        extended_private_train_query_types_counts = []
        extended_private_train_query_iterators = {}
        for query_type, query_answer_dict in train_private_data_dict.items():

            sub_query_types_dicts = separate_query_dict(query_answer_dict, nentity, nrelation)
            print("Number of queries of type of private queries " + query_type + ": " + str(len(sub_query_types_dicts)))

            for sub_query_type, sub_query_types_dict in sub_query_types_dicts.items():
                extended_private_train_query_types.append(sub_query_type)
                extended_private_train_query_types_counts.append(len(sub_query_types_dict))
                extended_private_train_data_dict[sub_query_type] = sub_query_types_dict

        extended_train_query_types_counts = np.array(extended_private_train_query_types_counts) / np.sum(
            extended_private_train_query_types_counts)

        print("Extended private query types: ", len(extended_private_train_query_types))

        for query_type in extended_private_train_query_types:
            query_answer_dict = extended_private_train_data_dict[query_type]
            print("====================================")
            print(query_type)

            new_iterator = SingledirectionalOneShotIterator(DataLoader(
                TrainDataset(nentity, nrelation, query_answer_dict,
                             baseline=True, nattribute=nattribute, value_vocab=value_vocab),
                batch_size=batch_size,
                shuffle=True,
                collate_fn=TrainDataset.collate_fn
            ))
            extended_private_train_query_iterators[query_type] = new_iterator

    # Add training iterators
    extended_train_data_dict = {}
    extended_train_query_types = []
    extended_train_query_types_counts = []
    extended_train_query_iterators = {}

    for query_type, query_answer_dict in train_data_dict.items():

        sub_query_types_dicts = separate_query_dict(query_answer_dict, nentity, nrelation)
        print("Number of queries of type " + query_type + ": " + str(len(sub_query_types_dicts)))

        for sub_query_type, sub_query_types_dict in sub_query_types_dicts.items():
            extended_train_query_types.append(sub_query_type)
            extended_train_query_types_counts.append(len(sub_query_types_dict))
            extended_train_data_dict[sub_query_type] = sub_query_types_dict

    extended_train_query_types_counts = np.array(extended_train_query_types_counts) / np.sum(
        extended_train_query_types_counts)

    print("Extended query types: ", len(extended_train_query_types))

    for query_type in extended_train_query_types:
        query_answer_dict = extended_train_data_dict[query_type]
        print("====================================")
        print(query_type)

        new_iterator = SingledirectionalOneShotIterator(DataLoader(
            TrainDataset(nentity, nrelation, query_answer_dict,
                         baseline=True, nattribute=nattribute, value_vocab=value_vocab),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=TrainDataset.collate_fn
        ))
        extended_train_query_iterators[query_type] = new_iterator

    print("====== Create Development Dataloader ======")
    baseline_validation_loaders = {}
    for query_type, query_answer_dict in valid_data_dict.items():

        sub_query_types_dicts = separate_query_dict(query_answer_dict, nentity, nrelation)

        for sub_query_type, sub_query_types_dict in sub_query_types_dicts.items():
            new_iterator = DataLoader(
                ValidDataset(nentity, nrelation, sub_query_types_dict,
                             baseline=True, nattribute=nattribute, value_vocab=value_vocab),
                batch_size=batch_size,
                shuffle=True,
                collate_fn=ValidDataset.collate_fn
            )
            baseline_validation_loaders[sub_query_type] = new_iterator

    print("====== Create Testing Dataloader ======")
    # combine test_private_answers and test_public_answers
    for query_type, query_answer_dict in test_data_dict.items():
        for single_query_answer in query_answer_dict.keys():
            test_data_dict[query_type][single_query_answer]["test_answers"] = \
                test_data_dict[query_type][single_query_answer]["test_private_answers"] + \
                test_data_dict[query_type][single_query_answer]["test_public_answers"]
            test_data_dict[query_type][single_query_answer]["test_private_answers"] = \
                test_data_dict[query_type][single_query_answer]["test_private_answers"] + \
                test_data_dict[query_type][single_query_answer]["valid_answers"]

    baseline_test_loaders = {}
    for query_type, query_answer_dict in test_data_dict.items():
        sub_query_types_dicts = separate_query_dict(query_answer_dict, nentity, nrelation)

        for sub_query_type, sub_query_types_dict in sub_query_types_dicts.items():
            new_iterator = DataLoader(
                TestDataset(nentity, nrelation, sub_query_types_dict,
                            baseline=True, nattribute=nattribute, value_vocab=value_vocab),
                batch_size=batch_size,
                shuffle=True,
                collate_fn=TestDataset.collate_fn
            )
            baseline_test_loaders[sub_query_type] = new_iterator

    if args.model == "q2p":
        model = Q2P(num_entities=nentity + nvalue,
                    num_relations=nrelation + nattribute * 2 + nnumerical_proj,
                    embedding_size=300)
    elif args.model == "gqe":
        model = GQE(num_entities=nentity + nvalue,
                    num_relations=nrelation + nattribute * 2 + nnumerical_proj,
                    embedding_size=300)

    elif args.model == "q2b":

        model = Q2B(num_entities=nentity + nvalue,
                    num_relations=nrelation + nattribute * 2 + nnumerical_proj,
                    embedding_size=300)

    else:
        raise NotImplementedError

    # TODO: QI HU
    if torch.cuda.is_available():
        model = model.cuda()

    step = 360000
    model_path = "../logs/" + args.model + "_" + str(step) + "_" + data_name + "_baseline.bin"
    model.load_state_dict(torch.load("../models/" + args.model + "_" + data_name + ".pt"))

    model.eval()

    all_implicit_generalization_logs = []
    all_implicit_generalization_logs_private = []
    all_implicit_generalization_logs_public = []
    abstracted_implicit_generalization_logs = {}
    abstracted_implicit_generalization_logs_private = {}
    abstracted_implicit_generalization_logs_public = {}
    print("====== Testing ======")
    for task_name, loader in baseline_test_loaders.items():
        all_generalization_logs = []
        all_generalization_logs_private = []
        all_generalization_logs_public = []
        for batched_query, unified_ids, train_answers, valid_answers, \
                test_answers, test_private_answers, test_public_answers in tqdm(loader):
            if args.model == "lstm" or args.model == "transformer":
                batched_query = unified_ids

            query_embedding = model(batched_query)
            generalization_logs = model.evaluate_generalization(query_embedding, valid_answers, test_answers)
            answer_sample_index = [True if len(item) > 0 else False for item in test_private_answers]
            if type(query_embedding) is tuple:
                query_embedding_sample = (query_embedding[0][answer_sample_index],
                                          query_embedding[1][answer_sample_index])
            else:
                query_embedding_sample = query_embedding[answer_sample_index]
            valid_answers_sample = [valid_answers[i] for i in range(len(valid_answers)) if
                                    answer_sample_index[i]]
            test_private_answers_sample = [test_private_answers[i] for i in range(len(test_private_answers))
                                           if answer_sample_index[i]]
            try:
                generalization_logs_private = model.evaluate_generalization(query_embedding_sample,
                                                                            valid_answers_sample,
                                                                            test_private_answers_sample)
                flag_private = True
            except:
                flag_private = False

            answer_sample_index = [True if len(item) > 0 else False for item in test_public_answers]
            if type(query_embedding) is tuple:
                query_embedding_sample = (query_embedding[0][answer_sample_index],
                                          query_embedding[1][answer_sample_index])
            else:
                query_embedding_sample = query_embedding[answer_sample_index]
            valid_answers_sample = [valid_answers[i] for i in range(len(valid_answers)) if
                                    answer_sample_index[i]]
            test_public_answers_sample = [test_public_answers[i] for i in range(len(test_public_answers))
                                          if answer_sample_index[i]]
            try:
                generalization_logs_public = model.evaluate_generalization(query_embedding,
                                                                           valid_answers_sample,
                                                                           test_public_answers_sample)
                flag_public = True
            except:
                flag_public = False

            all_generalization_logs.extend(generalization_logs)

            if loader.dataset.isImplicit:
                all_implicit_generalization_logs.extend(generalization_logs)
                abstract_query_type = loader.dataset.query_type
                # private
                if flag_private:
                    all_generalization_logs_private.extend(generalization_logs_private)
                    all_implicit_generalization_logs_private.extend(generalization_logs_private)
                # public
                if flag_public:
                    all_generalization_logs_public.extend(generalization_logs_public)
                    all_implicit_generalization_logs_public.extend(generalization_logs_public)

                if abstract_query_type in abstracted_implicit_generalization_logs:
                    # original
                    abstracted_implicit_generalization_logs[abstract_query_type].extend(generalization_logs)
                    # private
                    if flag_private:
                        try:
                            abstracted_implicit_generalization_logs_private[abstract_query_type].extend(
                                generalization_logs_private)
                        except:
                            abstracted_implicit_generalization_logs_private[abstract_query_type] = []
                            abstracted_implicit_generalization_logs_private[abstract_query_type].extend(
                                generalization_logs_private)
                    # public
                    if flag_public:
                        try:
                            abstracted_implicit_generalization_logs_public[abstract_query_type].extend(
                                generalization_logs_public)
                        except:
                            abstracted_implicit_generalization_logs_public[abstract_query_type] = []
                            abstracted_implicit_generalization_logs_public[abstract_query_type].extend(
                                generalization_logs_public)

                else:
                    # original
                    abstracted_implicit_generalization_logs[abstract_query_type] = []
                    abstracted_implicit_generalization_logs[abstract_query_type].extend(generalization_logs)
                    # private
                    if flag_private:
                        abstracted_implicit_generalization_logs_private[abstract_query_type] = []
                        abstracted_implicit_generalization_logs_private[abstract_query_type].extend(
                            generalization_logs_private)
                    # public
                    if flag_public:
                        abstracted_implicit_generalization_logs_public[abstract_query_type] = []
                        abstracted_implicit_generalization_logs_public[abstract_query_type].extend(
                            generalization_logs_public)


        aggregated_generalization_logs = log_aggregation(all_generalization_logs)
        aggregated_generalization_logs_private = log_aggregation(all_generalization_logs_private)
        aggregated_generalization_logs_public = log_aggregation(all_generalization_logs_public)

        for key, value in aggregated_generalization_logs.items():
            test_summary_writer.add_scalar("z-test-" + task_name + "-" + key, value, 0)

        for key, value in aggregated_generalization_logs_private.items():
            test_summary_writer.add_scalar("z-test-private-" + task_name + "-" + key, value, 0)

        for key, value in aggregated_generalization_logs_public.items():
            test_summary_writer.add_scalar("z-test-public-" + task_name + "-" + key, value, 0)

    aggregated_implicit_generalization_logs = log_aggregation(all_implicit_generalization_logs)

    aggregated_implicit_generalization_logs_private = log_aggregation(all_implicit_generalization_logs_private)
    aggregated_implicit_generalization_logs_public = log_aggregation(all_implicit_generalization_logs_public)

    for key, value in aggregated_implicit_generalization_logs.items():
        test_summary_writer.add_scalar("x-test-implicit-" + key, value, 0)

    for key, value in aggregated_implicit_generalization_logs_private.items():
        test_summary_writer.add_scalar("x-test-implicit-private-" + key, value, 0)

    for key, value in aggregated_implicit_generalization_logs_public.items():
        test_summary_writer.add_scalar("x-test-implicit-public-" + key, value, 0)

    for key, value in abstracted_implicit_generalization_logs.items():
        aggregated_value = log_aggregation(value)
        for metric, metric_value in aggregated_value.items():
            test_summary_writer.add_scalar("y-test-implicit-" + key + "-" + metric, metric_value, 0)

    for key, value in abstracted_implicit_generalization_logs_private.items():
        aggregated_value = log_aggregation(value)
        for metric, metric_value in aggregated_value.items():
            test_summary_writer.add_scalar("y-test-implicit-private-" + key + "-" + metric, metric_value, 0)

    for key, value in abstracted_implicit_generalization_logs_public.items():
        aggregated_value = log_aggregation(value)
        for metric, metric_value in aggregated_value.items():
            test_summary_writer.add_scalar("y-test-implicit-public-" + key + "-" + metric, metric_value, 0)

    gc.collect()


















