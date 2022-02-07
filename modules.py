"""
This model shows an example of using dgl.metapath_reachable_graph on the original heterogeneous
graph.Because the original HAN implementation only gives the preprocessed homogeneous graph, this mode
could not reproduce the result in HAN as they did not provide the preprocessing code, and we
constructed another dataset from ACM with a different set of papers, connections, features and
labels.
"""


import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import convert
from dgl.base import DGLError
from dgl.nn.functional import edge_softmax
from dgl.nn.pytorch import GATConv
from torch.autograd import Variable
from merge_edges import _merge_edge_weights


class AdvGATConv(GATConv):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True):
        super(AdvGATConv, self).__init__(in_feats, out_feats, num_heads, feat_drop,
                                         attn_drop, negative_slope, residual,
                                         activation, allow_zero_in_degree, bias)

    def forward(self, graph, node_feat, get_attention=False):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(node_feat, tuple):
                h_src = self.feat_drop(node_feat[0])
                h_dst = self.feat_drop(node_feat[1])
                if not hasattr(self, 'fc_src'):
                    feat_src = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
                    feat_dst = self.fc(h_dst).view(-1, self._num_heads, self._out_feats)
                else:
                    feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(node_feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))

            weight = graph.edata['weight']
            if weight.ndim is 1:
                weight = weight.view(-1, 1, 1)
            if weight.ndim is 2:
                weight = weight.unsqueeze(1)

            e = self.leaky_relu(graph.edata.pop('e') * weight)
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], self._num_heads, self._out_feats)
                rst = rst + resval
            # bias
            if self.bias is not None:
                rst = rst + self.bias.view(1, self._num_heads, self._out_feats)
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)
        return (beta * z).sum(1)  # (N, D * K)


def metapath_reachable_graph(g, metapath):
    adj = 1
    for etype in metapath:
        adj = adj * g.adj(etype=etype, scipy_fmt='csr', transpose=True)

    adj = (adj != 0).tocsr()
    srctype = g.to_canonical_etype(metapath[0])[0]
    dsttype = g.to_canonical_etype(metapath[-1])[2]
    src_nodes, dst_nodes = adj.nonzero()
    new_g = convert.heterograph({(srctype, '_E', dsttype): (src_nodes, dst_nodes)},
                                {srctype: adj.shape[0], dsttype: adj.shape[1]},
                                idtype=g.idtype, device=g.device)

    edge_weight_dict = _merge_edge_weights(g, metapath)
    edge_weights = []
    for src_node, dst_node in zip(src_nodes, dst_nodes):
        key = (src_node, dst_node)
        edge_weights.append(edge_weight_dict[key])
    edge_weights = torch.FloatTensor(edge_weights)
    new_g.edges[(srctype, '_E', dsttype)].data['weight'] = edge_weights

    # copy srcnode features
    new_g.nodes[srctype].data.update(g.nodes[srctype].data)
    # copy dstnode features
    if srctype != dsttype:
        new_g.nodes[dsttype].data.update(g.nodes[dsttype].data)
    return new_g


class HANLayer(nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    meta_paths : list of metapaths, each as a list of edge types
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    g : DGLHeteroGraph
        The heterogeneous graph
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """

    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()
        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()

        for i in range(len(meta_paths)):
            self.gat_layers.append(AdvGATConv(in_size, out_size, layer_num_heads,
                                              dropout, dropout, activation=F.elu,
                                              allow_zero_in_degree=True))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)
        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        semantic_embeddings = []
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = metapath_reachable_graph(g, meta_path)

        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)


class HanBlock(nn.Module):
    def __init__(self, meta_paths, in_size, hidden_size, out_size, num_heads, dropout):
        super(HanBlock, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(meta_paths, in_size, hidden_size, num_heads[0], dropout))

        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(meta_paths, hidden_size * num_heads[l - 1],
                                        hidden_size, num_heads[l], dropout))

        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)

        return self.predict(h)


class AdvHan(nn.Module):
    def __init__(self, hyper_param_wrt_ntypes):
        super(AdvHan, self).__init__()
        self.blocks = nn.ModuleList()
        for hyper_param in hyper_param_wrt_ntypes:
            block = HanBlock(*hyper_param)
            self.blocks.append(block)

    def forward(self, graph, h_wrt_ntypes):
        results = []
        for h, block in zip(h_wrt_ntypes, self.blocks):
            h = block(graph, h)
            results.append(h)

        return results


class NodePredictor(nn.Module):
    def __init__(self, feat_dim, hidden_dim, out_dim):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(feat_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, out_dim),
                                   nn.Softmax())

    def forward(self, h):
        return self.model(h)


class EdgePredictor(nn.Module):
    def __init__(self, src_feat_dim, dst_feat_dim, hidden_dim, out_dim):
        super().__init__()
        self.hidden = nn.Linear(src_feat_dim + dst_feat_dim, hidden_dim)
        self.out = nn.Sequential(nn.Linear(hidden_dim, out_dim), nn.Softmax())

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src['src_h'], edges.dst['dst_h']], -1)
        return {'score': self.out(F.relu(self.hidden(h))).squeeze(1)}

    def forward(self, g, src_h, dst_h):
        with g.local_scope():
            g.srcdata['src_h'] = src_h[g.srcnodes().long(), :]
            g.dstdata['dst_h'] = dst_h[g.dstnodes().long(), :]
            g.apply_edges(self.apply_edges)
            return g.edata['score']


class GMMEstNetwork(nn.Module):
    def __init__(self, latent_dim, h_dim, gamma_dim, dropout):
        super().__init__()
        self._calc_gamma = nn.Sequential(
            nn.Linear(latent_dim, h_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, gamma_dim),
            nn.Softmax()
        )

    @staticmethod
    def _calc_gmm_params(latent_samples, gamma):
        # gamma = N x K
        gamma_sum = gamma.sum(dim=0)

        # gamma_sum = K
        phi = gamma_sum / gamma.size(0)

        # mu = K x D
        mu = (gamma.unsqueeze(-1) * latent_samples.unsqueeze(1)).sum(dim=0) / gamma_sum.unsqueeze(-1)

        # latent_samples = N x D
        # z_mu = N x K x D
        z_mu = latent_samples.unsqueeze(1) - mu.unsqueeze(0)
        # z_mu_outer = N x K x D x D
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)

        # K x D x D
        sigma = (gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer).sum(dim=0) / gamma_sum.unsqueeze(-1).unsqueeze(-1)
        return phi, mu, sigma

    @staticmethod
    def _to_var(x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def _calc_energy(self, latent_samples, phi, mu, sigma):
        K, D, _ = sigma.size()

        z_mu = (latent_samples.unsqueeze(1) - mu.unsqueeze(0))

        sigma_inverse = []
        sigma_det = []
        sigma_diag = 0
        eps = 1e-12
        for i in range(K):
            # K x D x D
            sigma_k = sigma[i] + self._to_var(torch.eye(D) * eps)
            sigma_inverse.append(torch.inverse(sigma_k).unsqueeze(0))
            sigma_det.append((torch.cholesky(sigma_k.cpu() * (2 * np.pi)).diag().prod()).unsqueeze(0))
            sigma_diag = sigma_diag + torch.sum(1 / sigma_k.diag())

        # K x D x D
        cov_inverse = torch.cat(sigma_inverse, dim=0)
        # K
        sigma_det = torch.cat(sigma_det)

        # N x K
        exp_term_tmp = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)

        # for stability (logsumexp)
        max_val = torch.max(exp_term_tmp.clamp(min=0), dim=1, keepdim=True)[0]

        exp_term = torch.exp(exp_term_tmp - max_val)
        sample_energy = -max_val.squeeze() - torch.log(
            torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt(sigma_det)).unsqueeze(0), dim=1) + eps)

        return sample_energy, sigma_diag

    def forward(self, latent_samples, lambda_energy=0.2, lambda_sigma=0.02):
        gamma = self._calc_gamma(latent_samples)
        phi, mu, sigma = self._calc_gmm_params(latent_samples, gamma)
        sample_energy, sigma_diag = self._calc_energy(latent_samples, phi, mu, sigma)
        loss = lambda_energy * sample_energy + lambda_sigma * sigma_diag
        return loss
