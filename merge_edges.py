# encoding: utf-8

import numba as nb
import numpy as np
from collections import defaultdict


@nb.jit()
def mat_agg(mat_a, mat_b):
    n_row_a, n_col_a = mat_a.shape
    n_row_b, n_col_b = mat_b.shape
    ret_mat = np.zeros(shape=(n_row_a, n_col_b), dtype=np.float32)

    for i in range(n_row_a):
        for j in range(n_col_b):

            ret = 0.
            for k in range(n_col_a):
                elem_a = mat_a[i, k]
                elem_b = mat_b[k, j]
                ret += 0. if elem_a * elem_b == 0 else elem_a + elem_b

            ret_mat[i, j] = ret

    return ret_mat


def coordinate_to_mat(graph, meta_path, feat_name):
    src_type = graph.to_canonical_etype(meta_path)[0]
    dst_type = graph.to_canonical_etype(meta_path)[-1]

    feat_mat = np.zeros((graph.num_nodes(ntype=src_type), graph.num_nodes(ntype=dst_type)), dtype=np.float32)
    x, y = graph.edges(etype=meta_path)
    feat = graph.edges[meta_path].data[feat_name]
    feat_mat[x, y] = feat
    return feat_mat


def merge_edge_weights(graph, meta_paths, feat_name):
    path_length = len(meta_paths)
    if path_length is 1:
        return graph.edges[meta_paths[0]].data[feat_name]

    acc_feat_mat = coordinate_to_mat(graph, meta_paths[0], feat_name)
    for meta_index in range(1, path_length):
        cur_feat_mat = coordinate_to_mat(graph, meta_paths[meta_index], feat_name)
        acc_feat_mat = mat_agg(acc_feat_mat, cur_feat_mat)

    acc_feat_mat = acc_feat_mat / path_length
    return acc_feat_mat[acc_feat_mat > 0]


def gen_mapping(nodes_a, nodes_b, feats):
    merged_edges = defaultdict(list)
    for (src_node, dst_node, feat) in zip(nodes_a, nodes_b, feats):
        merged_edges[dst_node].append((src_node, feat))
    return merged_edges


def _merge_edge_weights(graph, meta_paths, feat_name='weight'):
    path_length = len(meta_paths)
    if path_length is 1:
        return graph.edges[meta_paths[0]].data[feat_name]

    src_nodes, dst_nodes = graph.edges(etype=meta_paths[0])
    src_nodes, dst_nodes = src_nodes.numpy(), dst_nodes.numpy()
    feats = graph.edges[meta_paths[0]].data[feat_name].numpy()
    acc_merged_edges = gen_mapping(src_nodes, dst_nodes, feats)

    for meta_index in range(1, path_length):
        src_nodes, dst_nodes = graph.edges(etype=meta_paths[meta_index])
        src_nodes, dst_nodes = src_nodes.numpy(), dst_nodes.numpy()
        feats = graph.edges[meta_paths[meta_index]].data[feat_name].numpy()
        cur_merged_edges = gen_mapping(dst_nodes, src_nodes, feats)

        join_nodes = set(acc_merged_edges.keys()) & set(cur_merged_edges.keys())

        # 更新聚合的边信息
        updated_merged_edges = defaultdict(list)
        for join_node in join_nodes:
            for (cur_node, cur_feat) in cur_merged_edges[join_node]:
                for (acc_node, acc_feat) in acc_merged_edges[join_node]:
                    updated_merged_edges[cur_node].append((acc_node, acc_feat + cur_feat))
        acc_merged_edges = updated_merged_edges

    # 聚合不同跳的边信息
    merged_edge_feats = defaultdict(list)
    for dst_node in acc_merged_edges:
        for (src_node, feat) in acc_merged_edges[dst_node]:
            merged_edge_feats[(src_node, dst_node)].append(feat / path_length)

    # 聚合不同路径的边信息
    results = {}
    for k in merged_edge_feats:
        results[k] = np.mean(merged_edge_feats[k])
    return results

