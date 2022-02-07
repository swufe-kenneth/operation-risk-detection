# encoding: utf-8

def is_abnormal(graph):
    in_degrees = graph.in_degrees(v=graph.nodes(ntype='pc'), etype='operate')
    in_degrees.apply_(lambda x: True if x >= 2 else False)
    graph.nodes['pc'].data['label'] = in_degrees

    out_degrees = graph.out_degrees(u=graph.nodes(ntype='staff'), etype='operate')
    out_degrees.apply_(lambda x: True if x >= 2 else False)
    graph.nodes['staff'].data['label'] = out_degrees
    return graph

