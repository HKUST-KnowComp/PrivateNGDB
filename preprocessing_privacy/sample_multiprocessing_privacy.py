from sample_privacy import *
import json
from multiprocessing import Pool

num_processes = 72


def sample_all_data(id):
    n_queries_train_dict = n_queries_train_dict_same
    n_queries_valid_test_dict = n_queries_valid_test_dict_same

    # first_round_query_types = {
    #     "1p": "(p,(e))",
    #     "2p": "(p,(p,(e)))",
    #     "2i": "(i,(p,(e)),(p,(e)))",
    #     "3i": "(i,(p,(e)),(p,(e)),(p,(e)))",
    #     "ip": "(p,(i,(p,(e)),(p,(e))))",
    #     "pi": "(i,(p,(p,(e))),(p,(e)))",
    #     "2u": "(u,(p,(e)),(p,(e)))",
    #     "up": "(p,(u,(p,(e)),(p,(e))))",
    #     "2in": "(i,(n,(p,(e))),(p,(e)))",
    #     "3in": "(i,(n,(p,(e))),(p,(e)),(p,(e)))",
    #     "inp": "(p,(i,(n,(p,(e))),(p,(e))))",
    #     "pin": "(i,(p,(p,(e))),(n,(p,(e))))",
    #     "pni": "(i,(n,(p,(p,(e)))),(p,(e)))",
    # }

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


    for experiment_number in ["e2", "e34"]:

        print("Experiment: " + experiment_number)
        for data_dir in n_queries_train_dict.keys():

            print("Load Train Graph " + data_dir)
            train_path = "../graphs/" + data_dir + "_" + experiment_number + "_train.pkl"
            train_graph = nx.read_gpickle(train_path)

            relation_edges_counter = 0
            attribute_edges_counter = 0
            reverse_attribute_edges_counter = 0
            numerical_edges_counter = 0
            for u, v, a in train_graph.edges(data=True):
                if isinstance(u, float) and isinstance(v, float):
                    numerical_edges_counter += len(a)
                elif isinstance(u, float):
                    reverse_attribute_edges_counter += len(a)
                elif isinstance(v, float):
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
            valid_path = "../graphs/" + data_dir + "_" + experiment_number + "_valid.pkl"
            valid_graph = nx.read_gpickle(valid_path)

            relation_edges_counter = 0
            attribute_edges_counter = 0
            reverse_attribute_edges_counter = 0
            numerical_edges_counter = 0
            for u, v, a in valid_graph.edges(data=True):
                if isinstance(u, float) and isinstance(v, float):
                    numerical_edges_counter += len(a)
                elif isinstance(u, float):
                    reverse_attribute_edges_counter += len(a)
                elif isinstance(v, float):
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
            test_path = "../graphs/" + data_dir + "_" + experiment_number + "_test.pkl"
            test_graph = nx.read_gpickle(test_path)

            relation_edges_counter = 0
            attribute_edges_counter = 0
            reverse_attribute_edges_counter = 0
            numerical_edges_counter = 0
            for u, v, a in test_graph.edges(data=True):
                if isinstance(u, float) and isinstance(v, float):
                    numerical_edges_counter += len(a)
                elif isinstance(u, float):
                    reverse_attribute_edges_counter += len(a)
                elif isinstance(v, float):
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
                if isinstance(u, float) and isinstance(v, float):
                    numerical_edges_counter += len(a)
                    print("example numerical edge: ", u, v, a)
                    break

            for u, v, a in test_graph.edges(data=True):
                if isinstance(u, float) and isinstance(v, str):
                    reverse_attribute_edges_counter += len(a)
                    print("example reverse attribute edge: ", u, v, a)
                    break

            for u, v, a in test_graph.edges(data=True):
                if isinstance(v, float) and isinstance(u, str):
                    attribute_edges_counter += len(a)
                    print("example attribute edge: ", u, v, a)
                    break

            for u, v, a in test_graph.edges(data=True):
                if isinstance(v, str) and isinstance(u, str):
                    attribute_edges_counter += len(a)
                    print("example relation edge: ", u, v, a)
                    break

            for u, v, a in test_graph.edges(data=True):
                if isinstance(u, float) and isinstance(v, float):
                    l = test_graph[float(str(u))][float(str(v))]

            if experiment_number == "e2":
                train_graph_sampler = GraphSamplerE2(train_graph)
                valid_graph_sampler = GraphSamplerE2(valid_graph)
                test_graph_sampler = GraphSamplerE2(test_graph)

            elif experiment_number == "e34":
                train_graph_sampler = GraphSamplerE34(train_graph)
                valid_graph_sampler = GraphSamplerE34(valid_graph)
                test_graph_sampler = GraphSamplerE34(test_graph)

            print("sample training queries")

            train_queries = {}

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

                    if len(valid_query_train_answers) > 0 and len(valid_query_valid_answers) > 0 \
                            and len(valid_query_train_answers) != len(valid_query_valid_answers):
                        break

                return sampled_valid_query, valid_query_train_answers, valid_query_valid_answers

            def sample_test_graph_with_pattern(pattern):
                while True:

                    sampled_test_query = test_graph_sampler.sample_with_pattern(pattern)

                    test_query_train_answers = train_graph_sampler.query_search_answer(sampled_test_query)
                    test_query_valid_answers = valid_graph_sampler.query_search_answer(sampled_test_query)
                    test_query_test_answers = test_graph_sampler.query_search_answer(sampled_test_query)

                    if len(test_query_train_answers) > 0 and len(test_query_valid_answers) > 0 \
                            and len(test_query_test_answers) > 0 and len(test_query_test_answers) != len(
                        test_query_valid_answers):
                        break
                return sampled_test_query, test_query_train_answers, test_query_valid_answers, test_query_test_answers

            for query_type, sample_pattern in first_round_query_types.items():

                print("train query_type: ", query_type)

                this_type_train_queries = {}

                n_query = n_queries_train_dict[data_dir] // num_processes

                for _ in tqdm(range(n_query)):
                    sampled_train_query, train_query_train_answers = sample_train_graph_with_pattern(sample_pattern)
                    this_type_train_queries[sampled_train_query] = {"train_answers": train_query_train_answers}

                train_queries[sample_pattern] = this_type_train_queries

            with open(
                    "../sampled_data/" + data_dir + "_" + experiment_number + "_train_queries_" + str(id) + ".json",
                    "w") as file_handle:
                json.dump(train_queries, file_handle)

            print("sample validation queries")

            validation_queries = {}
            for query_type, sample_pattern in first_round_query_types.items():
                print("validation query_type: ", query_type)

                this_type_validation_queries = {}

                n_query = n_queries_valid_test_dict[data_dir] // num_processes

                if query_type == "1p":
                    n_query = n_query * 10

                for _ in tqdm(range(n_query)):
                    sampled_valid_query, valid_query_train_answers, valid_query_valid_answers = \
                        sample_valid_graph_with_pattern(sample_pattern)

                    this_type_validation_queries[sampled_valid_query] = {
                        "train_answers": valid_query_train_answers,
                        "valid_answers": valid_query_valid_answers
                    }

                validation_queries[sample_pattern] = this_type_validation_queries

            with open(
                    "../sampled_data/" + data_dir + "_" + experiment_number + "_valid_queries_" + str(id) + ".json",
                    "w") as file_handle:
                json.dump(validation_queries, file_handle)

            print("sample testing queries")

            test_queries = {}

            for query_type, sample_pattern in first_round_query_types.items():
                print("test query_type: ", query_type)
                this_type_test_queries = {}

                n_query = n_queries_valid_test_dict[data_dir] // num_processes

                if query_type == "1p":
                    n_query = n_query * 10

                for _ in tqdm(range(n_query)):
                    sampled_test_query, test_query_train_answers, test_query_valid_answers, test_query_test_answers = \
                        sample_test_graph_with_pattern(sample_pattern)

                    this_type_test_queries[sampled_test_query] = {
                        "train_answers": test_query_train_answers,
                        "valid_answers": test_query_valid_answers,
                        "test_answers": test_query_test_answers
                    }

                test_queries[sample_pattern] = this_type_test_queries
            with open(
                    "../sampled_data/" + data_dir + "_" + experiment_number + "_test_queries_" + str(id) + ".json",
                    "w") as file_handle:
                json.dump(test_queries, file_handle)


if __name__ == "__main__":
    with Pool(num_processes) as p:
        print(p.map(sample_all_data, range(num_processes)))
