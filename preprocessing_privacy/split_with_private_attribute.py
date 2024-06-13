import networkx as nx

from random import sample, choice, randint, shuffle
import csv

from datetime import datetime as dt
from collections import Counter
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import json
import pickle

# split validation set and test set for privacy evaluation

def split_private_queries(queries):
    # check whether the query is in the train graph
    private_queries = {}
    public_queries = {}

    for query_type, query_answer_dict in queries.items():
        private_queries[query_type] = {}
        for query, answer in query_answer_dict.items():
            need_to_inferred_answer = list(set(answer['valid_answers']) - set(answer['train_answers']))
            if len(need_to_inferred_answer) > 0:
                need_to_inferred_answer = shuffle(need_to_inferred_answer)
                split_index = len(need_to_inferred_answer) // 2
                private = need_to_inferred_answer[:split_index]
                public = need_to_inferred_answer[split_index:]
                private_queries[query_type][query] = private + answer['train_answers']
                public_queries[query_type][query] = public + answer['train_answers']
            else:
                private_queries[query_type][query] = answer['train_answers']
                public_queries[query_type][query] = answer['train_answers']

    return private_queries, public_queries


def sample_private_1p(num_1p, valid_graph, test_graph):
    private_1p = []
    private_1p_reverse = []
    edge_list = []
    for u, v in test_graph.edges():
        if isinstance(u, str) and isinstance(v, tuple):
            edge_list.append((u, v))

    for u, v in edge_list:
        valid_graph_attr = valid_graph.get_edge_data(u, v)
        test_graph_attr = test_graph.get_edge_data(u, v)
        if valid_graph_attr != test_graph_attr:
            try:
                diff_attr = list(set(test_graph_attr) - set(valid_graph_attr))
            except:
                diff_attr = list(set(test_graph_attr))
            sub_attr = {key: value for key, value in test_graph_attr.items() if key in diff_attr}
            private_1p.append((u, v, sub_attr))

    private_1p_sample = sample(private_1p, num_1p)
    private_1p_attr = []
    for u, v, attr in private_1p_sample:
        if len(attr) > 1:
            attr = choice(dict(attr.keys()))
            attr = {attr: test_graph.get_edge_data(u, v)[attr]}
        private_1p_attr.append((u, v, attr))

    for u, v, attr in private_1p_attr:
        attr = {key+1: value for key, value in attr.items()}
        private_1p_reverse.append((v, u, attr))
    private_1p = private_1p_attr + private_1p_reverse
    return private_1p


if __name__ == '__main__':
    # load train graph
    data_name = ['FB15K', 'DB15K', 'YAGO15K']
    for dataset in data_name:
        # with open('./' + dataset + '_small_train_with_units.pkl', 'rb') as f:
        #     train_graph = pickle.load(f)

        with open('../preprocessing/' + dataset + '_small_valid_with_units.pkl', 'rb') as f:
            valid_graph = pickle.load(f)
        with open('../preprocessing/' + dataset + '_small_test_with_units.pkl', 'rb') as f:
            test_graph = pickle.load(f)

        # sample private 1p attribute list (attribute and reverse attribute)
        attr_list = sample_private_1p(10, valid_graph, test_graph)
        # split private queries
        # private_quires, public_quires = split_private_queries_with_1p(valid_queries, attr_list)



        # save private queries
        # with open('../input_files_small/' + dataset + '_small_valid_private_queries.pkl', 'wb') as f:
        #     pickle.dump(valid_private_quires, f)
        # with open('../input_files_small/' + dataset + '_small_valid_public_queries.pkl', 'wb') as f:
        #     pickle.dump(valid_public_quires, f)

        # test_private_quires = {}
        # test_public_quires = {}
        # for query_type, query_answer_dict in test_queries.items():
        #     test_private_quires[query_type] = {}
        #     test_public_quires[query_type] = {}
        #     for query, answer in query_answer_dict.items():
        #         if isinstance(answer['valid_answers'][0], list):
        #             set1 = set(map(tuple, answer['test_answers']))
        #             set2 = set(map(tuple, answer['valid_answers']))
        #             need_to_inferred_answer = [list(sublist) for sublist in (set1 - set2)]
        #         else:
        #             need_to_inferred_answer = list(set(answer['test_answers']) - set(answer['valid_answers']))
        #         if len(need_to_inferred_answer) > 0:
        #             shuffle(need_to_inferred_answer)
        #             split_index = len(need_to_inferred_answer) // 2
        #             private = need_to_inferred_answer[:split_index]
        #             public = need_to_inferred_answer[split_index:]
        #             test_private_quires[query_type][query] = private + answer['valid_answers']
        #             test_public_quires[query_type][query] = public + answer['valid_answers']
        #         else:
        #             test_private_quires[query_type][query] = answer['valid_answers']
        #             test_public_quires[query_type][query] = answer['valid_answers']



