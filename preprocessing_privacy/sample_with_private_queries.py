from multiprocessing import Pool
from sample_with_numeral_1p_privacy import *
import json
import random

num_processes = 72

n_queries_train_dict_same = {
    "FB15K": 300000 // 20,
    "DB15K": 200000 // 20,
    "YAGO15K": 200000 // 20
}

def sample_private_queries(id):
    n_queries_train_dict = n_queries_train_dict_same

    first_round_query_types = {
        "1p": "(p,(e))",
        "2p": "(p,(p,(e)))",
        "2i": "(i,(p,(e)),(p,(e)))",
        "3i": "(i,(p,(e)),(p,(e)),(p,(e)))",
        "ip": "(p,(i,(p,(e)),(p,(e))))",
        "pi": "(i,(p,(p,(e))),(p,(e)))",
        "2u": "(u,(p,(e)),(p,(e)))",
        "up": "(p,(u,(p,(e)),(p,(e))))",
    }

    for data_dir in n_queries_train_dict.keys():

        print("Load Train Graph " + data_dir)
        train_path = "./" + data_dir + "_train_with_units.pkl"
        # train_graph = nx.read_gpickle(train_path)

        with open(train_path, "rb") as file_handle:
            train_graph = pickle.load(file_handle)

        relation_edges_counter = 0
        attribute_edges_counter = 0
        reverse_attribute_edges_counter = 0
        numerical_edges_counter = 0
        for u, v, a in train_graph.edges(data=True):
            if isinstance(u, tuple) and isinstance(v, tuple):
                numerical_edges_counter += len(a)
            elif isinstance(u, tuple):
                reverse_attribute_edges_counter += len(a)
            elif isinstance(v, tuple):
                attribute_edges_counter += len(a)
            elif isinstance(u, str) and isinstance(v, str):
                relation_edges_counter += len(a)

        print("#nodes: ", train_graph.number_of_nodes())
        print("#relation edges: ", relation_edges_counter)
        print("#attribute edges: ", attribute_edges_counter)
        print("#reverse attribute edges: ", reverse_attribute_edges_counter)
        print("#numerical edges: ", numerical_edges_counter)
        print("#all edges: ", relation_edges_counter + attribute_edges_counter +
              reverse_attribute_edges_counter + numerical_edges_counter)

        # load private 1p queries
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
        train_graph.add_edges_from(private_1p)

        train_graph_sampler = GraphSamplerE34(train_graph)

        def sample_train_graph_with_pattern_with_private_1p(pattern, private_1p):
            nodes = [item[0] for item in private_1p]
            i = 0
            while True:
                node = random.choice(nodes)
                sampled_train_query = train_graph_sampler.sample_with_pattern_on_node(pattern, node)
                if sampled_train_query is None:
                    return None, None
                train_query_train_answers, private_flag = train_graph_sampler.query_search_answer_with_private_attr(sampled_train_query, private_1p)
                if len(train_query_train_answers) > 0 and private_flag:
                    break
                i += 1
                if i > 100:
                    return None, None
            return sampled_train_query, train_query_train_answers


        print("sample training queries")
        train_queries = {}
        for query_type, sample_pattern in first_round_query_types.items():

            print("train query_type: ", query_type)

            this_type_train_queries = {}

            n_query = n_queries_train_dict[data_dir] // num_processes

            for _ in tqdm(range(n_query)):
                sampled_train_query, train_query_train_answers = sample_train_graph_with_pattern_with_private_1p(sample_pattern, private_1p)
                # negative_query_train_answers = sample_negative_query(sampled_train_query)
                if sampled_train_query is None:
                    break
                this_type_train_queries[sampled_train_query] = {"train_answers": train_query_train_answers}


            train_queries[sample_pattern] = this_type_train_queries

        with open(
                "../sampled_data_privacy_query/" + data_dir + "_train_private_queries_" + str(id) + "_with_units.json",
                "w") as file_handle:
            json.dump(train_queries, file_handle)

if __name__ == "__main__":

    with Pool(num_processes) as p:
        print(p.map(sample_private_queries, range(num_processes)))
    # num_processes = 1
    # sample_private_queries(1)