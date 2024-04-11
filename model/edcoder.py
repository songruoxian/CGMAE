from typing import Optional
from itertools import chain
from functools import partial

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from .gat import GAT
from .gcn import GCN
from .loss_func import sce_loss
from utils import create_norm, drop_edge


def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead,
                 nhead_out, attn_drop, negative_slope=0.2, concat_out=True) -> nn.Module:
    if m_type == "gat":
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gcn":
        mod = GCN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding")
        )
    elif m_type == "mlp":
        mod = nn.Sequential(
            nn.Linear(in_dim, num_hidden),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden, out_dim)
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError

    return mod

class Node_Alignment_Choose(nn.Module):
    def __init__(self, nodes=150):
        super().__init__()
        self.nodes = nodes
        self.e2e = nn.Sequential(
            E2E(1, 8, (nodes, nodes),nodes),
            nn.LeakyReLU(0.33),
            E2E(8, 8, (nodes, nodes),nodes),
            nn.LeakyReLU(0.33),
        )

        self.e2n = nn.Sequential(
            nn.Conv2d(8, 256, (1, nodes)),
            nn.LeakyReLU(0.33),
        )
    def forward(self, A):
        x = self.e2e(A)
        x = self.e2n(x)
        x = x.reshape(self.nodes,-1)
        return x

class E2E(nn.Module):

    def __init__(self, in_channel, out_channel, input_shape, nodes,**kwargs):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.d = input_shape[0]
        self.conv1xd = nn.Conv2d(in_channel, out_channel, (self.d, 1))
        self.convdx1 = nn.Conv2d(in_channel, out_channel, (1, self.d))
        self.nodes = nodes

    def forward(self, A):
        A = A.view(1, self.in_channel, self.nodes, self.nodes)
        a = self.conv1xd(A)
        b = self.convdx1(A)

        concat1 = torch.cat([a] * self.d, 2)
        concat2 = torch.cat([b] * self.d, 3)
        return concat1 + concat2

class PreModel(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            mask_rate: float = 0.3,
            encoder_type: str = "gat",
            decoder_type: str = "gat",
            loss_fn: str = "sce",
            drop_edge_rate: float = 0.0,
            replace_rate: float = 0.1,
            alpha_l: float = 2,
            concat_hidden: bool = False,
            classes: int = 5,
    ):
        super(PreModel, self).__init__()
        self._mask_rate = mask_rate

        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._drop_edge_rate = drop_edge_rate
        self._output_hidden_size = num_hidden
        self._concat_hidden = concat_hidden

        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate
        self._classes = classes

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead_out if decoder_type in ("gat", "dotgat") else num_hidden

        # build decoder for attribute prediction
        self.decoder_s = setup_module(
            m_type=decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=in_dim,
            num_layers=1,
            nhead=nhead,
            nhead_out=nhead_out,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=True,
        )
        self.decoder_t = setup_module(
            m_type=decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=in_dim,
            num_layers=1,
            nhead=nhead,
            nhead_out=nhead_out,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=True,
        )

        # build cls model for source classification
        self.cls_model = nn.Sequential(
            nn.Linear(enc_nhead * enc_num_hidden, classes),
        )

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        self.E2EN = Node_Alignment_Choose(nodes=20)
        self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)
        self.cls_loss = nn.CrossEntropyLoss()
        self.domain_loss = nn.NLLLoss()
        self.node_loss = nn.MSELoss()

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def aggregate_subgraph_to_graph(self, g, x):
        g_ = g.clone()
        g_.ndata['feat'] = x
        walks = dgl.sampling.random_walk(g_, torch.arange(g_.number_of_nodes()).cuda(), length=50)
        for i in range(len(walks[0])):
            if g_.out_degree(walks[0][i][0]) == 0:
                continue
            else:
                walks[0][i] = torch.masked_fill(walks[0][i], torch.eq(walks[0][i], -1), walks[0][i][0])
                sub_graph = dgl.node_subgraph(g_, walks[0][i])
                output_node = dgl.mean_nodes(sub_graph, 'feat')
                g_.nodes[walks[0][i][0]].data['feat'] = output_node

        return g_

    def E2EN_subgraph_to_graph(self, g, x):
        g_e = g.to('cuda')
        g_e.ndata['feat'] = x.cuda()
        g_attr = g.to('cuda')
        g_attr.ndata['feat'] = x.cuda()
        walks = dgl.sampling.random_walk(g_e.to('cpu'), torch.arange(g_e.number_of_nodes()).to('cpu'), length=30,restart_prob=0)

        for i in range(len(walks[0])):
            if g_e.out_degree(walks[0][i][0].to('cuda')) == 0:
                continue
            else:
                walks[0][i] = torch.masked_fill(walks[0][i], torch.eq(walks[0][i], -1), walks[0][i][0])
                only_random_walk = walks[0][i][1:]
                sub_graph_e = dgl.node_subgraph(g_e, only_random_walk.to('cuda'))
                sub_graph_attr = dgl.node_subgraph(g_e, only_random_walk.to('cuda'))
                A_e = sub_graph_e.adjacency_matrix()
                A_e_= A_e.to_dense().cuda()
                x = self.E2EN(A_e_)
                sub_graph_e.ndata['feat'] = x
                output_node_e = dgl.mean_nodes(sub_graph_e, 'feat')
                g_e.nodes[walks[0][i][0].to('cuda')].data['feat'] = output_node_e
                output_node_attr = dgl.mean_nodes(sub_graph_attr, 'feat')
                g_attr.nodes[walks[0][i][0].to('cuda')].data['feat'] = output_node_attr
        return g_e, g_attr

    def forward(self, g_t, x_t, g_s, x_s, y_s):
        # ---- edge reconstruction ----
        loss = self.mask_edge_prediction(g_t, x_t, g_s, x_s,y_s)
        loss_item = {"loss": loss.item()}
        return loss, loss_item

    def decode_edge(self, rec_x):
        x_hat = torch.sigmoid(torch.matmul(rec_x, rec_x.t()))
        return x_hat

    def node_alignment(self, use_g_t,rep_t,g_s,rep_s):
        align_g = use_g_t.clone()
        align_g.ndata['feat'] = rep_t

        '''node_alignment_E2E_E2N'''
        sub_graph_target_e, sub_graph_target_attr= self.E2EN_subgraph_to_graph(use_g_t, rep_t)
        sub_graph_source_e, sub_graph_source_attr= self.E2EN_subgraph_to_graph(g_s, rep_s)
        idx_e = self.calculate_two_similary(sub_graph_source_e, sub_graph_target_e)
        idx_attr = self.calculate_two_similary(sub_graph_source_attr, sub_graph_target_attr)
        one_hot_idx_e = torch.nn.functional.one_hot(idx_e, num_classes=g_s.number_of_nodes()).float()
        one_hot_idx_attr = torch.nn.functional.one_hot(idx_attr, num_classes=g_s.number_of_nodes()).float()
        loss_e = self.node_loss(one_hot_idx_e, one_hot_idx_attr)
        loss_attr = self.node_loss(one_hot_idx_attr, one_hot_idx_e)
        loss_node = loss_e + loss_attr
        print('idx_e: ',idx_e)
        align_g.nodes[torch.arange(align_g.number_of_nodes()).cuda()].data['feat'] = rep_s[idx_e]
        feature = align_g.ndata['feat']
        A = self.decode_edge(feature)
        return A,loss_node

    def calculate_two_similary(self, g1, g2):
        x_s = g1.ndata['feat']
        x_t = g2.ndata['feat']
        graph_attention = torch.mm(x_t, x_s.t())
        print('graph_attention: ',graph_attention.shape)
        a_x = nn.Softmax(dim=1)(graph_attention)  # i->j
        a_y = nn.Softmax(dim=0)(graph_attention)  # j->i
        max_values, idx = torch.max(a_x, dim=1)
        return idx

    # our model
    def mask_edge_prediction(self, g_t, x_t, g_s, x_s, y_s):
        if self._drop_edge_rate > 0:
            use_g_t, masked_edges = drop_edge(g_t, self._drop_edge_rate, return_edges=True)
        else:
            use_g_t = g_t

        enc_rep_t, all_hidden_t = self.encoder(use_g_t, x_t, return_hidden=True)
        enc_rep_s, all_hidden_s = self.encoder(g_s, x_s, return_hidden=True)
        if self._concat_hidden:
            enc_rep_t = torch.cat(all_hidden_t, dim=1)
            enc_rep_s = torch.cat(all_hidden_s, dim=1)

        rep_t = self.encoder_to_decoder(enc_rep_t)
        rep_s = self.encoder_to_decoder(enc_rep_s)

        if self._decoder_type in ("mlp", "linear"):
            recon_t = self.decoder_t(rep_t)
            recon_s = self.decoder_s(rep_s)
        else:
            recon_t = self.decoder_t(use_g_t, rep_t)
            recon_s = self.decoder_s(g_s, rep_s)

        '''node alignment'''
        rec_adj_t, loss_node = self.node_alignment(g_t,rep_t,g_s,rep_s)
        mask_adj_t = use_g_t.adjacency_matrix()
        mask_adj_t_ = mask_adj_t.to_dense().cuda()
        rec_adj_t_ = rec_adj_t * mask_adj_t_
        loss_edge = self.criterion(rec_adj_t_, mask_adj_t_)
        loss_attr_t = self.criterion(recon_t, x_t)
        loss_attr_s = self.criterion(recon_s, x_s)

        '''node reconstruction and edge reconstruction for target domain'''
        loss_rec =  loss_edge + loss_attr_s + loss_attr_t

        '''compute node classification loss for S '''
        source_logits = self.cls_model(rep_s)
        train_mask = g_s.ndata["train_mask"]
        loss_source_node = self.cls_loss(source_logits[train_mask], y_s[train_mask])

        ''' compute entropy loss for T '''
        target_probs = self.cls_model(rep_t)
        target_probs = torch.clamp(target_probs, min=1e-9, max=1.0)
        loss_entropy = torch.mean(torch.sum(-target_probs * torch.log(target_probs), dim=-1))

        loss = loss_rec + loss_entropy + loss_source_node + loss_node

        return loss

    def embed(self, g, x):
        rep = self.encoder(g, x)
        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()

    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
