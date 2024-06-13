from sample_with_numeral_1p_privacy import *
import json
from multiprocessing import Pool

num_processes = 72

n_private_1p_attribute = {
    "FB15K": 4000,
    "DB15K": 3000,
    "YAGO15K": 800
}

def sample_private_1p(num_1p, train_graph, valid_graph, test_graph):
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

    node_in_graph = []
    for p in private_1p:
        node = p[1]
        if train_graph.has_node(node):
            node_in_graph.append(True)
        else:
            node_in_graph.append(False)
    private_1p = [p for p, flag in zip(private_1p, node_in_graph) if flag]
    private_1p_sample = sample(private_1p, num_1p)
    private_1p_attr = []
    for u, v, attr in private_1p_sample:
        if len(attr) > 1:
            attr = choice(list(attr.keys()))
            attr = {attr: test_graph.get_edge_data(u, v)[attr]}
        private_1p_attr.append((u, v, attr))

    for u, v, attr in private_1p_attr:
        attr = {key + 1: value for key, value in attr.items()}
        private_1p_reverse.append((v, u, attr))
    private_1p = private_1p_attr + private_1p_reverse
    return private_1p

private_1p_dict = {}

for data_dir in n_queries_train_dict_same.keys():
    print("Load Train Graph " + data_dir)
    train_path = "./" + data_dir + "_train_with_units.pkl"
    with open(train_path, "rb") as file_handle:
        train_graph = pickle.load(file_handle)

    print("Load Valid Graph " + data_dir)
    valid_path = "./" + data_dir + "_valid_with_units.pkl"
    with open(valid_path, "rb") as file_handle:
        valid_graph = pickle.load(file_handle)

    print("Load Test Graph " + data_dir)
    test_path = "./" + data_dir +  "_test_with_units.pkl"
    with open(test_path, "rb") as file_handle:
        test_graph = pickle.load(file_handle)

    # random select private 1p attributes
    num_private_1p = n_private_1p_attribute[data_dir]
    private_1p_dir = sample_private_1p(num_private_1p, train_graph, valid_graph, test_graph)
    with open(
            "../input_files_privacy/" + data_dir + "_private_1p_queries.json",
            "w") as file_handle:
        json.dump(private_1p_dir, file_handle)
    private_1p_dict[data_dir] = private_1p_dir


def sample_all_data(id):
    n_queries_train_dict = n_queries_train_dict_same
    n_queries_valid_test_dict = n_queries_valid_test_dict_same


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

        print("Load Valid Graph " + data_dir)
        valid_path = "./" + data_dir +  "_valid_with_units.pkl"
        # valid_graph = nx.read_gpickle(valid_path)
        with open(valid_path, "rb") as file_handle:
            valid_graph = pickle.load(file_handle)

        relation_edges_counter = 0
        attribute_edges_counter = 0
        reverse_attribute_edges_counter = 0
        numerical_edges_counter = 0
        for u, v, a in valid_graph.edges(data=True):
            if isinstance(u, tuple) and isinstance(v, tuple):
                numerical_edges_counter += len(a)
            elif isinstance(u, tuple):
                reverse_attribute_edges_counter += len(a)
            elif isinstance(v, tuple):
                attribute_edges_counter += len(a)
            elif isinstance(u, str) and isinstance(v, str):
                relation_edges_counter += len(a)

        print("number of nodes: ", len(valid_graph.nodes))

        print("#relation edges: ", relation_edges_counter)
        print("#attribute edges: ", attribute_edges_counter)
        print("#reverse attribute edges: ", reverse_attribute_edges_counter)
        print("#numerical edges: ", numerical_edges_counter)
        print("#all edges: ", relation_edges_counter + attribute_edges_counter +
                reverse_attribute_edges_counter + numerical_edges_counter)

        print("Load Test Graph " + data_dir)
        test_path = "./" + data_dir + "_test_with_units.pkl"
        # test_graph = nx.read_gpickle(test_path)
        with open(test_path, "rb") as file_handle:
            test_graph = pickle.load(file_handle)

        relation_edges_counter = 0
        attribute_edges_counter = 0
        reverse_attribute_edges_counter = 0
        numerical_edges_counter = 0
        for u, v, a in test_graph.edges(data=True):
            if isinstance(u, tuple) and isinstance(v, tuple):
                numerical_edges_counter += len(a)
            elif isinstance(u, tuple):
                reverse_attribute_edges_counter += len(a)
            elif isinstance(v, tuple):
                attribute_edges_counter += len(a)
            elif isinstance(u, str) and isinstance(v, str):
                relation_edges_counter += len(a)

        print("number of nodes: ", len(test_graph.nodes))
        print("#relation edges: ", relation_edges_counter)
        print("#attribute edges: ", attribute_edges_counter)
        print("#reverse attribute edges: ", reverse_attribute_edges_counter)
        print("#numerical edges: ", numerical_edges_counter)
        print("#all edges: ", relation_edges_counter + attribute_edges_counter +
                reverse_attribute_edges_counter + numerical_edges_counter)

        # Print example edges:
        for u, v, a in test_graph.edges(data=True):
            if isinstance(u, tuple) and isinstance(v, tuple):
                print("example numerical edge: ", u, v, a)
                break

        for u, v, a in test_graph.edges(data=True):
            if isinstance(u, tuple) and isinstance(v, str):
                print("example reverse attribute edge: ", u, v, a)
                break

        for u, v, a in test_graph.edges(data=True):
            if isinstance(v, tuple) and isinstance(u, str):
                print("example attribute edge: ", u, v, a)
                break

        for u, v, a in test_graph.edges(data=True):
            if isinstance(v, str) and isinstance(u, str):
                print("example relation edge: ", u, v, a)
                break

        private_1p = private_1p_dict[data_dir]

        train_graph_sampler = GraphSamplerE34(train_graph)
        valid_graph_sampler = GraphSamplerE34(valid_graph)
        test_graph_sampler = GraphSamplerE34(test_graph)

        def sample_train_graph_with_pattern(pattern):
            while True:

                sampled_train_query = train_graph_sampler.sample_with_pattern(pattern)

                train_query_train_answers = train_graph_sampler.query_search_answer(sampled_train_query)
                if len(train_query_train_answers) > 0:
                    break
            return sampled_train_query, train_query_train_answers

        def sample_valid_graph_with_pattern(pattern):
            while True:

                sampled_valid_query = valid_graph_sampler.sample_with_pattern(pattern)

                valid_query_train_answers = train_graph_sampler.query_search_answer(sampled_valid_query)
                valid_query_valid_answers = valid_graph_sampler.query_search_answer(sampled_valid_query)

                if len(valid_query_valid_answers) > 0 \
                        and len(valid_query_train_answers) != len(valid_query_valid_answers):
                    break

            return sampled_valid_query, valid_query_train_answers, valid_query_valid_answers

        def sample_test_graph_with_pattern(pattern):
            while True:

                sampled_test_query = test_graph_sampler.sample_with_pattern(pattern)
                test_query_train_answers = train_graph_sampler.query_search_answer(sampled_test_query)
                test_query_valid_answers = valid_graph_sampler.query_search_answer(sampled_test_query)
                test_query_test_answers = test_graph_sampler.query_search_answer(sampled_test_query)

                if len(test_query_test_answers) > 0 and len(test_query_test_answers) != len(
                    test_query_valid_answers):
                    break
            return sampled_test_query, test_query_train_answers, test_query_valid_answers, test_query_test_answers

        # sample test graph query and split to public answer and private answer according to private 1p attributes
        def sample_test_graph_with_pattern_with_private_1p(pattern, private_1p):
            while True:

                sampled_test_query = test_graph_sampler.sample_with_pattern(pattern)

                test_query_train_answers = train_graph_sampler.query_search_answer(sampled_test_query)
                test_query_valid_answers = valid_graph_sampler.query_search_answer(sampled_test_query)
                test_query_test_answers, private_flag = test_graph_sampler.query_search_answer_with_private_attr(sampled_test_query, private_1p)
                if len(test_query_test_answers) > 0 and len(test_query_test_answers) != len(
                    test_query_valid_answers):
                    break
            return sampled_test_query, test_query_train_answers, test_query_valid_answers, test_query_test_answers, private_flag


        print("sample training queries")
        train_queries = {}

        for query_type, sample_pattern in first_round_query_types.items():

            print("train query_type: ", query_type)

            this_type_train_queries = {}

            n_query = n_queries_train_dict[data_dir] // num_processes

            for _ in tqdm(range(n_query)):
                sampled_train_query, train_query_train_answers = sample_train_graph_with_pattern(sample_pattern)
                this_type_train_queries[sampled_train_query] = {"train_answers": train_query_train_answers}

            train_queries[sample_pattern] = this_type_train_queries

        with open(
                "../sampled_data_privacy/" + data_dir + "_train_queries_" + str(id) + "_with_units.json",
                "w") as file_handle:
            json.dump(train_queries, file_handle)

        print("sample validation queries")

        validation_queries = {}
        for query_type, sample_pattern in first_round_query_types.items():
            print("validation query_type: ", query_type)

            this_type_validation_queries = {}

            n_query = n_queries_valid_test_dict[data_dir] // num_processes



            for _ in tqdm(range(n_query)):
                sampled_valid_query, valid_query_train_answers, valid_query_valid_answers = \
                    sample_valid_graph_with_pattern(sample_pattern)

                this_type_validation_queries[sampled_valid_query] = {
                    "train_answers": valid_query_train_answers,
                    "valid_answers": valid_query_valid_answers
                }

            validation_queries[sample_pattern] = this_type_validation_queries

        with open(
                "../sampled_data_privacy/" + data_dir  + "_valid_queries_" + str(id) + "_with_units.json",
                "w") as file_handle:
            json.dump(validation_queries, file_handle)

        print("sample testing queries")

        test_queries = {}
        test_private_public_queries = {}

        for query_type, sample_pattern in first_round_query_types.items():
            print("test query_type: ", query_type)
            this_type_test_queries = {}
            this_type_test_private_public_queries = {}
            n_query = n_queries_valid_test_dict[data_dir] // num_processes

            
            for _ in tqdm(range(n_query)):
                sampled_test_query, test_query_train_answers, \
                    test_query_valid_answers, test_query_test_answers, private_flag = \
                    sample_test_graph_with_pattern_with_private_1p(sample_pattern, private_1p)

                this_type_test_queries[sampled_test_query] = {
                    "train_answers": test_query_train_answers,
                    "valid_answers": test_query_valid_answers,
                    "test_answers": test_query_test_answers
                }
                test_query_test_private_answers = [answer for answer in test_query_test_answers if private_flag[answer]]
                test_query_test_public_answers = list(set(test_query_test_answers) - set(test_query_test_private_answers))
                this_type_test_private_public_queries[sampled_test_query] = {
                    "train_answers": test_query_train_answers,
                    "valid_answers": test_query_valid_answers,
                    "test_private_answers": test_query_test_private_answers,
                    "test_public_answers": test_query_test_public_answers
                }
                # if any(private_flag.values()):
                #     print(test_query_test_private_answers)
            test_queries[sample_pattern] = this_type_test_queries
            test_private_public_queries[sample_pattern] = this_type_test_private_public_queries
        with open(
                "../sampled_data_privacy/" + data_dir +  "_test_queries_" + str(id) + "_with_units.json",
                "w") as file_handle:
            json.dump(test_queries, file_handle)
        with open(
                "../sampled_data_privacy/" + data_dir +  "_test_private_public_queries_" + str(id) + "_with_units.json",
                "w") as file_handle:
            json.dump(test_private_public_queries, file_handle)


if __name__ == "__main__":

    with Pool(num_processes) as p:
        print(p.map(sample_all_data, range(num_processes)))
    # sample_all_data(1)
