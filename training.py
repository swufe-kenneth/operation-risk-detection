# encoding: utf-8

from preprocessing import clean_sequential_data
from utils import trans_to_hetero, drop_duplicated_edges, gen_graph_pairs
from embedding_model import MultipleGraphEmbeddingModel
import pickle as pkl
from os.path import join

filename = ''
original_datasets = clean_sequential_data(filename)

graph_datasets = {}
for date in original_datasets:
    graph_dataset = original_datasets[date]
    graph_dataset = drop_duplicated_edges(graph_dataset)
    hetero_graph = trans_to_hetero(graph_dataset)
    pos_neg_graph_pairs = gen_graph_pairs(hetero_graph)

    graph_datasets[date] = (hetero_graph, pos_neg_graph_pairs)

multiple_graph_embedding_model = MultipleGraphEmbeddingModel()
multiple_graph_embedding_model.train(graph_datasets.values())

embedding_rets = {}
for date in graph_datasets:
    embedding = multiple_graph_embedding_model.gen_embeddings(graph_datasets[date])
    embedding_rets[date] = embedding

prefix = ''
with open(join(prefix, 'original_datasets.pkl'), 'wb') as fw:
    pkl.dump(original_datasets, fw)

with open(join(prefix, 'graph_datasets.pkl'), 'wb') as fw:
    pkl.dump(graph_datasets, fw)

with open(join(prefix, 'embedding_rets.pkl'), 'wb') as fw:
    pkl.dump(embedding_rets, fw)
