# encoding: utf-8

import torch

from modules import AdvHan, EdgePredictor, NodePredictor
from settings import SRC_EDGE_DST_TYPES


class MultipleGraphEmbeddingModel(object):

    def __init__(self, hidden_dim=8,
                 embedding_dim=2,
                 num_heads=[8, 8],
                 dropout=0.2,
                 lr=0.005,
                 weight_decay=0.001,
                 epochs=500,
                 alpha=0.8,
                 staff_feat_dim=7,
                 dept_feat_dim=3,
                 pc_feat_dim=6,
                 sys_feat_dim=13,
                 node_labels=4,
                 edge_labels=2,
                 ):
        self.epochs = epochs
        self.alpha = alpha

        # define meta paths
        staff_meta_paths = [['affiliated_with', 'affiliated_with_'],
                            ['operate', 'operate_']]

        dept_meta_paths = [['affiliated_with_', 'affiliated_with']]

        pc_meta_paths = [['operate_', 'operate'],
                         ['query', 'query_'],
                         ['update', 'update_'],
                         ['add', 'add_'],
                         ['delete', 'delete_'],
                         ['download', 'download_']]

        sys_meta_paths = [['query_', 'query'],
                          ['update_', 'update'],
                          ['add_', 'add'],
                          ['delete_', 'delete'],
                          ['download_', 'download']]
        # initialize the model
        # aggregate the main parameters and define the embedding model
        params = [(staff_meta_paths, staff_feat_dim, hidden_dim, embedding_dim, num_heads, dropout),
                  (dept_meta_paths, dept_feat_dim, hidden_dim, embedding_dim, num_heads, dropout),
                  (pc_meta_paths, pc_feat_dim, hidden_dim, embedding_dim, num_heads, dropout),
                  (sys_meta_paths, sys_feat_dim, hidden_dim, embedding_dim, num_heads, dropout)]
        self.embedding_model = AdvHan(params)

        # define the prediction model for node label
        self.node_label_predictor = NodePredictor(embedding_dim, embedding_dim, node_labels)
        models = [self.embedding_model, self.node_label_predictor]

        # define the prediction model for graph structure
        self.structure_predictor = EdgePredictor(embedding_dim, embedding_dim, embedding_dim, edge_labels)
        models.append(self.structure_predictor)

        # define the classification loss and the optimizer.
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam([{'params': m.parameters()} for m in models], lr=lr,
                                          weight_decay=weight_decay)

    def train(self, data_sets):
        for epoch in range(self.epochs):
            for graph, graph_pairs in data_sets:
                # define the node features for every node types
                staff_feat = graph.nodes['staff'].data['staff_char']
                dept_feat = graph.nodes['department'].data['outsource']
                pc_feat = torch.cat((graph.nodes['pc'].data['is_inner_ip'], graph.nodes['pc'].data['is_wifi']), dim=-1)
                sys_feat = graph.nodes['system'].data['sys_cate']
                node_feat = [staff_feat, dept_feat, pc_feat, sys_feat]

                # define the label for every node types
                node_label_true = torch.LongTensor([0] * graph.num_nodes(ntype='staff') +
                                                   [1] * graph.num_nodes(ntype='department') +
                                                   [2] * graph.num_nodes(ntype='pc') +
                                                   [3] * graph.num_nodes(ntype='system'))

                # predict the node embedding
                embeddings = self.embedding_model(graph, node_feat)

                # predict the node label
                node_prob_pred = self.node_label_predictor(torch.cat(embeddings, dim=0))
                node_loss = self.loss_fn(node_prob_pred, node_label_true)

                # predict the graph structure for every relation types
                staff_embedding, dept_embedding, pc_embedding, sys_embedding = embeddings
                embedding_pairs = {
                    SRC_EDGE_DST_TYPES.AFF_REL: (staff_embedding, dept_embedding),
                    SRC_EDGE_DST_TYPES.OP_REL: (staff_embedding, pc_embedding),
                    SRC_EDGE_DST_TYPES.QUERY_REL: (pc_embedding, sys_embedding),
                    SRC_EDGE_DST_TYPES.UPDATE_REL: (pc_embedding, sys_embedding),
                    SRC_EDGE_DST_TYPES.ADD_REL: (pc_embedding, sys_embedding),
                    SRC_EDGE_DST_TYPES.DEL_REL: (pc_embedding, sys_embedding),
                    SRC_EDGE_DST_TYPES.DL_REL: (pc_embedding, sys_embedding)}

                structure_loss = 0.

                for rel_type in embedding_pairs:
                    pos_graph, neg_graph = graph_pairs[rel_type]
                    pos_scores = self.structure_predictor(pos_graph, *embedding_pairs[rel_type])
                    neg_scores = self.structure_predictor(neg_graph, *embedding_pairs[rel_type])
                    scores = torch.cat([pos_scores, neg_scores])
                    labels = torch.cat([torch.ones(pos_scores.size(0), dtype=torch.int64),
                                        torch.zeros(neg_scores.size(0), dtype=torch.int64)])

                    loss_per_relation = self.loss_fn(scores, labels)
                    if not torch.isnan(loss_per_relation):
                        structure_loss += loss_per_relation
                total_loss = self.alpha * structure_loss + (1 - self.alpha) * node_loss

                print('epoch: {}, node_loss: {:.4f}, '
                      'structure_loss: {:.4f}, total_loss: {:.4f}'.format(epoch + 1, node_loss, structure_loss,
                                                                          total_loss))

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

    def gen_embeddings(self, data_set):
        graph, graph_pairs = data_set

        # define the node features for every node types
        staff_feat = graph.nodes['staff'].data['staff_char']
        dept_feat = graph.nodes['department'].data['outsource']
        pc_feat = torch.cat((graph.nodes['pc'].data['is_inner_ip'], graph.nodes['pc'].data['is_wifi']), dim=-1)
        sys_feat = graph.nodes['system'].data['sys_cate']
        node_feat = [staff_feat, dept_feat, pc_feat, sys_feat]

        embeddings = self.embedding_model(graph, node_feat)
        return embeddings
