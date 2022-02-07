# encoding: utf-8
# (src_type, edge_type, dst_type) specifying the source node, edge, and destination node types.


class SRC_EDGE_DST_TYPES(object):
    AFF_REL = ('staff', 'affiliated_with', 'department')
    AFF_REL_ = ('department', 'affiliated_with_', 'staff')
    OP_REL = ('staff', 'operate', 'pc')
    OP_REL_ = ('pc', 'operate_', 'staff')
    QUERY_REL = ('pc', 'query', 'system')
    QUERY_REL_ = ('system', 'query_', 'pc')
    UPDATE_REL = ('pc', 'update', 'system')
    UPDATE_REL_ = ('system', 'update_', 'pc')
    ADD_REL = ('pc', 'add', 'system')
    ADD_REL_ = ('system', 'add_', 'pc')
    DEL_REL = ('pc', 'delete', 'system')
    DEL_REL_ = ('system', 'delete_', 'pc')
    DL_REL = ('pc', 'download', 'system')
    DL_REL_ = ('system', 'download_', 'pc')
