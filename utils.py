# encoding:utf-8
import dgl
import numpy as np
import torch

from settings import SRC_EDGE_DST_TYPES

AFF_REL = SRC_EDGE_DST_TYPES.AFF_REL
AFF_REL_ = SRC_EDGE_DST_TYPES.AFF_REL_
OP_REL = SRC_EDGE_DST_TYPES.OP_REL
OP_REL_ = SRC_EDGE_DST_TYPES.OP_REL_
QUERY_REL = SRC_EDGE_DST_TYPES.QUERY_REL
QUERY_REL_ = SRC_EDGE_DST_TYPES.QUERY_REL_
UPDATE_REL = SRC_EDGE_DST_TYPES.UPDATE_REL
UPDATE_REL_ = SRC_EDGE_DST_TYPES.UPDATE_REL_
ADD_REL = SRC_EDGE_DST_TYPES.ADD_REL
ADD_REL_ = SRC_EDGE_DST_TYPES.ADD_REL_
DEL_REL = SRC_EDGE_DST_TYPES.DEL_REL
DEL_REL_ = SRC_EDGE_DST_TYPES.DEL_REL_
DL_REL = SRC_EDGE_DST_TYPES.DL_REL
DL_REL_ = SRC_EDGE_DST_TYPES.DL_REL_

REL_TYPES = [AFF_REL, AFF_REL_, OP_REL, OP_REL_, QUERY_REL, QUERY_REL_, UPDATE_REL, UPDATE_REL_,
             ADD_REL, ADD_REL_, DEL_REL, DEL_REL_, DL_REL, DL_REL_]


class ToGraphData(object):
    def __init__(self):
        self.dgl_graph = {k: ([], []) for k in REL_TYPES}
        self.created_staff_nodes = {
            'max_nid': -1
        }
        self.created_dept_nodes = {
            'max_nid': -1
        }
        self.created_pc_nodes = {
            'max_nid': -1
        }
        self.created_sys_nodes = {
            'max_nid': -1
        }
        self.edge_feats = {k: [] for k in REL_TYPES}

    @staticmethod
    def _create_nid(node_name, feats, created_nodes):
        if node_name in created_nodes:
            nid = created_nodes[node_name][0]
            return nid, created_nodes

        nid = created_nodes['max_nid'] + 1
        created_nodes[node_name] = (nid, feats)
        created_nodes['max_nid'] = nid
        return nid, created_nodes

    def __call__(self, agg_data):
        agg_data.index = range(agg_data.shape[0])
        for index in agg_data.index:
            # create the nodes and features for department
            dept_name = agg_data.loc[index, 'dept_name']
            feats = agg_data.loc[index, 'is_outsource_dept']
            dept_nid, self.created_dept_nodes = self._create_nid(dept_name, feats, self.created_dept_nodes)
            self.dgl_graph[AFF_REL][1].append(dept_nid)
            self.dgl_graph[AFF_REL_][0].append(dept_nid)

            # create the nodes and features for staff
            employ_num = agg_data.loc[index, 'emply_num']
            feats = [agg_data.loc[index, 'acct_num'], agg_data.loc[index, 'operdr_empl_char_cd'],
                     agg_data.loc[index, 'emply_char_name']]
            staff_nid, self.created_staff_nodes = self._create_nid(employ_num, feats, self.created_staff_nodes)

            self.dgl_graph[AFF_REL][0].append(staff_nid)
            self.dgl_graph[AFF_REL_][1].append(staff_nid)
            self.edge_feats[AFF_REL].append(1)
            self.edge_feats[AFF_REL_].append(1)

            self.dgl_graph[OP_REL][0].append(staff_nid)
            self.dgl_graph[OP_REL_][1].append(staff_nid)

            # create the nodes and features for pc
            ip_addr = agg_data.loc[index, 'src_ip_addr']
            feats = [agg_data.loc[index, 'is_inner_ip'], agg_data.loc[index, 'is_wifi']]
            pc_nid, self.created_pc_nodes = self._create_nid(ip_addr, feats, self.created_pc_nodes)
            self.dgl_graph[OP_REL][1].append(pc_nid)
            self.dgl_graph[OP_REL_][0].append(pc_nid)
            self.edge_feats[OP_REL].append(1)
            self.edge_feats[OP_REL_].append(1)

            # create the nodes and features for system
            sys_name = agg_data.loc[index, 'sys_name']
            feats = [agg_data.loc[index, 'access_res'], 1 if agg_data.loc[index, 'operdr_cust_ind'] == '是' else 0]
            sys_nid, self.created_sys_nodes = self._create_nid(sys_name, feats, self.created_sys_nodes)

            behav_cate = agg_data.loc[index, 'behav_cate_clean']
            behav_counts = agg_data.loc[index, 'behavior_counts']
            if behav_cate == 'query':
                self.dgl_graph[QUERY_REL][0].append(pc_nid)
                self.dgl_graph[QUERY_REL][1].append(sys_nid)
                self.edge_feats[QUERY_REL].append(behav_counts)

                self.dgl_graph[QUERY_REL_][1].append(pc_nid)
                self.dgl_graph[QUERY_REL_][0].append(sys_nid)
                self.edge_feats[QUERY_REL_].append(behav_counts)
            elif behav_cate == 'update':
                self.dgl_graph[UPDATE_REL][0].append(pc_nid)
                self.dgl_graph[UPDATE_REL][1].append(sys_nid)
                self.edge_feats[UPDATE_REL].append(behav_counts)

                self.dgl_graph[UPDATE_REL_][1].append(pc_nid)
                self.dgl_graph[UPDATE_REL_][0].append(sys_nid)
                self.edge_feats[UPDATE_REL_].append(behav_counts)
            elif behav_cate == 'add':
                self.dgl_graph[ADD_REL][0].append(pc_nid)
                self.dgl_graph[ADD_REL][1].append(sys_nid)
                self.edge_feats[ADD_REL].append(behav_counts)

                self.dgl_graph[ADD_REL_][1].append(pc_nid)
                self.dgl_graph[ADD_REL_][0].append(sys_nid)
                self.edge_feats[ADD_REL_].append(behav_counts)
            elif behav_cate == 'delete':
                self.dgl_graph[DEL_REL][0].append(pc_nid)
                self.dgl_graph[DEL_REL][1].append(sys_nid)
                self.edge_feats[DEL_REL].append(behav_counts)

                self.dgl_graph[DEL_REL_][1].append(pc_nid)
                self.dgl_graph[DEL_REL_][0].append(sys_nid)
                self.edge_feats[DEL_REL_].append(behav_counts)
            else:
                self.dgl_graph[DL_REL][0].append(pc_nid)
                self.dgl_graph[DL_REL][1].append(sys_nid)
                self.edge_feats[DL_REL].append(behav_counts)

                self.dgl_graph[DL_REL_][1].append(pc_nid)
                self.dgl_graph[DL_REL_][0].append(sys_nid)
                self.edge_feats[DL_REL_].append(behav_counts)


def drop_duplicated_edges(graph_data):
    for rel_type in graph_data.dgl_graph:
        src_nodes, dst_nodes = graph_data.dgl_graph[rel_type]

        unique_set = set()
        unique_src_nodes = []
        unique_dst_nodes = []
        unique_edge_feats = []
        for (index, (src_node, dst_node)) in enumerate(zip(src_nodes, dst_nodes)):
            if (src_node, dst_node) not in unique_set:
                unique_src_nodes.append(src_node)
                unique_dst_nodes.append(dst_node)
                unique_edge_feats.append(graph_data.edge_feats[rel_type][index])

                unique_set.add((src_node, dst_node))
        graph_data.dgl_graph[rel_type] = (unique_src_nodes, unique_dst_nodes)
        graph_data.edge_feats[rel_type] = unique_edge_feats
    return graph_data


def to_onehot(data, categories):
    if type(categories).__name__ != 'list':
        raise TypeError('categories expects list, but {}'.format(type(categories).__name__))

    if 'ELSE' not in categories:
        categories.append('ELSE')

    onehot_dict = np.eye(len(categories), dtype=np.float32)
    cate_onehot_mapper = {c: onehot_dict[i, :] for (i, c) in enumerate(categories)}

    onehot_mat = []
    for d in data:
        onehot_vec = cate_onehot_mapper.get(d)
        if onehot_vec is None:
            onehot_vec = cate_onehot_mapper['ELSE']
        onehot_mat.append(onehot_vec)
    return torch.from_numpy(np.vstack(onehot_mat))


def staff_feat_to_onehot(data):
    categories = ['人力外包', '暑期实习生', '项目外包', '派遣员工', '行编', '校招实习生']
    return to_onehot(data, categories)


def binary_to_onehot(data):
    categories = [0, 1]
    return to_onehot(data, categories)


def sys_feat_to_onehot(data):
    categories = ['分布式消费信贷核心系统（生产+ 测试）',
                  '操作平台',
                  '统一业务运营管理平台（生产+测试）',
                  '统一业务运营管理平台（生产+测试)',
                  '统一认证',
                  '般若',
                  '综合数据查询系统',
                  '发欺诈天启',
                  '资信平台',
                  '综合资金管理系统（生产+ 测试）',
                  '玄明',
                  '电子档案管理子系统']
    return to_onehot(data, categories)


def trans_to_hetero(graph_data):
    graph = dgl.heterograph(graph_data.dgl_graph, idtype=torch.int32)

    if 'max_nid' in graph_data.created_staff_nodes:
        graph_data.created_staff_nodes.pop('max_nid')
    feats = [v[1][2] for v in graph_data.created_staff_nodes.values()]
    graph.nodes['staff'].data['staff_char'] = staff_feat_to_onehot(feats)

    if 'max_nid' in graph_data.created_dept_nodes:
        graph_data.created_dept_nodes.pop('max_nid')
    feats = [v[1] for v in graph_data.created_dept_nodes.values()]
    graph.nodes['department'].data['outsource'] = binary_to_onehot(feats)

    if 'max_nid' in graph_data.created_pc_nodes:
        graph_data.created_pc_nodes.pop('max_nid')
    feats = [v[1][0] for v in graph_data.created_pc_nodes.values()]
    graph.nodes['pc'].data['is_inner_ip'] = binary_to_onehot(feats)

    feats = [v[1][1] for v in graph_data.created_pc_nodes.values()]
    graph.nodes['pc'].data['is_wifi'] = binary_to_onehot(feats)

    if 'max_nid' in graph_data.created_sys_nodes:
        graph_data.created_sys_nodes.pop('max_nid')
    feats = list(graph_data.created_sys_nodes.keys())
    graph.nodes['system'].data['sys_cate'] = sys_feat_to_onehot(feats)

    graph.edges[AFF_REL].data['weight'] = torch.tensor(graph_data.edge_feats[AFF_REL], dtype=torch.float32)
    graph.edges[OP_REL].data['weight'] = torch.tensor(graph_data.edge_feats[OP_REL], dtype=torch.float32)
    graph.edges[QUERY_REL].data['weight'] = torch.tensor(graph_data.edge_feats[QUERY_REL], dtype=torch.float32)
    graph.edges[UPDATE_REL].data['weight'] = torch.tensor(graph_data.edge_feats[UPDATE_REL], dtype=torch.float32)
    graph.edges[ADD_REL].data['weight'] = torch.tensor(graph_data.edge_feats[ADD_REL], dtype=torch.float32)
    graph.edges[DEL_REL].data['weight'] = torch.tensor(graph_data.edge_feats[DEL_REL], dtype=torch.float32)
    graph.edges[DL_REL].data['weight'] = torch.tensor(graph_data.edge_feats[DL_REL], dtype=torch.float32)

    graph.edges[AFF_REL_].data['weight'] = torch.tensor(graph_data.edge_feats[AFF_REL], dtype=torch.float32)
    graph.edges[OP_REL_].data['weight'] = torch.tensor(graph_data.edge_feats[OP_REL], dtype=torch.float32)
    graph.edges[QUERY_REL_].data['weight'] = torch.tensor(graph_data.edge_feats[QUERY_REL], dtype=torch.float32)
    graph.edges[UPDATE_REL_].data['weight'] = torch.tensor(graph_data.edge_feats[UPDATE_REL], dtype=torch.float32)
    graph.edges[ADD_REL_].data['weight'] = torch.tensor(graph_data.edge_feats[ADD_REL], dtype=torch.float32)
    graph.edges[DEL_REL_].data['weight'] = torch.tensor(graph_data.edge_feats[DEL_REL], dtype=torch.float32)
    graph.edges[DL_REL_].data['weight'] = torch.tensor(graph_data.edge_feats[DL_REL], dtype=torch.float32)

    return graph


def gen_graph_pairs(graph):
    meta_paths = [AFF_REL, OP_REL, QUERY_REL, UPDATE_REL, ADD_REL, DEL_REL, DL_REL]

    graphs_pairs = {}
    for (ntype_src, etype, ntype_dst) in meta_paths:
        node_src, node_dst = graph.edges(etype=etype)
        adj_mat = np.zeros((graph.num_src_nodes(ntype=ntype_src), graph.num_dst_nodes(ntype=ntype_dst)),
                           dtype=np.float32)
        adj_mat[node_src.numpy(), node_dst.numpy()] = 1.0
        neg_src, neg_dst = np.where(adj_mat != 1.0)
        index = np.random.permutation(range(neg_src.shape[0]))[:node_src.size(0)]

        sampled_neg_src = torch.from_numpy(neg_src[index])
        sampled_neg_dst = torch.from_numpy(neg_dst[index])

        neg_graph = dgl.heterograph({(ntype_src, etype, ntype_dst): (sampled_neg_src, sampled_neg_dst)})
        pos_graph = graph.edge_type_subgraph(etypes=[etype])
        graphs_pairs[(ntype_src, etype, ntype_dst)] = [pos_graph, neg_graph]
    return graphs_pairs
