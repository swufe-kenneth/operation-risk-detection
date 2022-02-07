# encoding: utf-8

import numpy as np
from py2neo import Graph, Node, Relationship

from settings import SRC_EDGE_DST_TYPES


class Connect2DB(object):
    def __init__(self, timestamp):
        self._graph = Graph('bolt://10.82.199.46:7687', auth=('neo4j', 'Tesla&123'))
        self._created_nodes = {}
        self._timestamp = timestamp

    def _create_node(self, label, **attr):
        for k in attr:
            if isinstance(attr[k], (int, float, str, bytes, bool)):
                continue

            if np.isnan(attr[k]):
                attr[k] = None
                continue

            if 'int' in type(attr[k]).__name__:
                attr[k] = int(attr[k])

            if 'float' in type(attr[k]).__name__:
                attr[k] = float(attr[k])

        node = Node(label, timestamp=self._timestamp, **attr)
        self._graph.create(node)
        return node

    def _safe_create_node(self, node_id, label, **attr):
        if node_id not in self._created_nodes:
            node = self._create_node(label, **attr)
            self._created_nodes[node_id] = node
            return node
        else:
            return self._created_nodes[node_id]

    def create_dept_node(self, dept_name, **attr):
        return self._safe_create_node(node_id=dept_name, label='department',
                                      dept_name=dept_name, **attr)

    def create_staff_node(self, staff_uid, **attr):
        return self._safe_create_node(node_id=staff_uid, label='staff', staff_name=staff_uid, **attr)

    def create_pc_node(self, ip_addr, **attr):
        return self._safe_create_node(node_id=ip_addr, label='pc', ip_addr=ip_addr, **attr)

    def create_sys_node(self, sys_name, **attr):
        return self._safe_create_node(node_id=sys_name, label='sys', sys_name=sys_name, **attr)

    def create_edge(self, src_node, edge_type, dst_node, **attr):
        edge = Relationship(src_node, edge_type, dst_node)
        for k in attr:
            if isinstance(attr[k], (int, float, str, bytes, bool)):
                edge[k] = attr[k]

            if np.isnan(attr[k]):
                edge[k] = None
                continue

            if 'int' in type(attr[k]).__name__:
                edge[k] = int(attr[k])

            if 'float' in type(attr[k]).__name__:
                edge[k] = float(attr[k])

        self._graph.create(edge)


AFF_REL = SRC_EDGE_DST_TYPES.AFF_REL
OP_REL = SRC_EDGE_DST_TYPES.OP_REL
QUERY_REL = SRC_EDGE_DST_TYPES.QUERY_REL
UPDATE_REL = SRC_EDGE_DST_TYPES.UPDATE_REL
ADD_REL = SRC_EDGE_DST_TYPES.ADD_REL
DEL_REL = SRC_EDGE_DST_TYPES.DEL_REL
DL_REL = SRC_EDGE_DST_TYPES.DL_REL
REL_TYPES = [AFF_REL, OP_REL, QUERY_REL, UPDATE_REL, ADD_REL, DEL_REL, DL_REL]


class Dump2DB(object):
    def __init__(self, timestamp, graph_data):
        self._connect2db = Connect2DB(timestamp)
        self._graph_data = graph_data
        self._dept_nodes = {}
        self._staff_nodes = {}
        self._pc_nodes = {}
        self._sys_nodes = {}

    def _create_nodes(self, node_type):
        node_data_dicts = getattr(self._graph_data, node_type)
        for main_id in node_data_dicts:
            if main_id == 'max_nid':
                continue
            sec_id, node_feats = node_data_dicts[main_id]
            yield main_id, sec_id, node_feats

    def _create_dept_nodes(self):
        for (dept_name, node_id, is_outsource) in self._create_nodes('created_dept_nodes'):
            node = self._connect2db.create_dept_node(dept_name=dept_name, is_outsource=is_outsource)
            self._dept_nodes[node_id] = node

    def _create_staff_nodes(self):
        for (staff_uid, node_id, feats) in self._create_nodes('created_staff_nodes'):
            node = self._connect2db.create_staff_node(staff_uid=staff_uid,
                                                      account=feats[0],
                                                      staff_category=feats[2])
            self._staff_nodes[node_id] = node

    def _create_pc_nodes(self):
        for (ip_addr, node_id, feats) in self._create_nodes('created_pc_nodes'):
            node = self._connect2db.create_pc_node(ip_addr=ip_addr,
                                                   is_inner_addr=feats[0],
                                                   is_wifi=feats[1])
            self._pc_nodes[node_id] = node

    def _create_sys_nodes(self):
        for (sys_name, node_id, feats) in self._create_nodes('created_sys_nodes'):
            node = self._connect2db.create_sys_node(sys_name=sys_name,
                                                    accessed=feats[0])
            self._sys_nodes[node_id] = node

    def _create_edges(self):
        for rel_type in REL_TYPES:
            src_ids, dst_ids = self._graph_data.dgl_graph[rel_type]
            edge_feats = self._graph_data.edge_feats[rel_type]

            for (src_id, dst_id, edge_feat) in zip(src_ids, dst_ids, edge_feats):
                edge_type = rel_type[1]

                if rel_type == AFF_REL:
                    src_node = self._staff_nodes[src_id]
                    dst_node = self._dept_nodes[dst_id]
                    self._connect2db.create_edge(src_node, edge_type, dst_node, weight=edge_feat)

                elif rel_type == OP_REL:
                    src_node = self._staff_nodes[src_id]
                    dst_node = self._pc_nodes[dst_id]
                    self._connect2db.create_edge(src_node, edge_type, dst_node, weight=edge_feat)

                else:
                    src_node = self._pc_nodes[src_id]
                    dst_node = self._sys_nodes[dst_id]
                    self._connect2db.create_edge(src_node, edge_type, dst_node, weight=edge_feat)

    def __call__(self):
        self._create_dept_nodes()
        self._create_staff_nodes()
        self._create_pc_nodes()
        self._create_sys_nodes()
        self._create_edges()
