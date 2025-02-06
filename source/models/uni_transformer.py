"""
update:
add bond type diffusion
add electrostatic feature
update shape feature
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_softmax, scatter_sum, scatter_mean
from torch_geometric.nn import radius_graph, knn_graph
from utils.covalent_graph import connect_covalent_graph
import utils.data as utils_data
import time
import copy
from models.common import *
from models.shape_vn_layers import VNLinear, VNStdFeature, VNLinearLeakyReLU
import pdb
from utils.analyze import construct_bond_tensors

EPS = 1e-6

class BaseGVPAttLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, shape_dim, n_heads, edge_feat_dim, r_feat_dim, 
                 mess_gvp_layer_num=3, node_gvp_layer_num=3, act_fn='relu', use_shape_vec_mul=True, 
                 use_residue=True, mlp_norm=True, output_norm=True, l_idx=0):
        super().__init__()
        self.input_dim = input_dim # (scalar_dim, vector_dim)
        self.hidden_dim = hidden_dim
        self.shape_dim = shape_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.act_fn = act_fn
        self.use_shape_vec_mul = use_shape_vec_mul
        self.use_residue = use_residue
        self.edge_feat_dim = edge_feat_dim
        self.r_feat_dim = r_feat_dim
        self.output_norm = output_norm
        #self.time_emb_dim = time_emb_dim

        # init shape embedding layer to combine atom embedding with shape embedding
        shape_emb_dim = input_dim[0] + shape_dim * input_dim[1] + shape_dim + input_dim[1]
        self.shape_in = VNLinear(shape_dim, shape_dim)
        self.shape_emb_layer = nn.Sequential(
                                nn.Linear(shape_emb_dim, hidden_dim[0]),
                                nn.ReLU(),
                                nn.Linear(hidden_dim[0], hidden_dim[0])
                            )
        self.shape_scalar_layer = MLP(input_dim[0] + hidden_dim[0], hidden_dim[0], hidden_dim[0], norm=mlp_norm, act_fn=act_fn)

        # init vector shape embedding to integrate shape embedding into vector features
        if self.use_shape_vec_mul:
            self.shape_vec_layer = nn.Sequential(
                                    nn.Linear(hidden_dim[0], 2 * hidden_dim[0]),
                                    nn.ReLU(),
                                    nn.Linear(2 * hidden_dim[0], shape_dim * input_dim[1]),
                                )
            self.vector_emb_shape_layer = VNLinearLeakyReLU(input_dim[1] * 2, hidden_dim[1], dim=3)
        else:
            self.vector_emb_shape_layer = VNLinearLeakyReLU(input_dim[1] + shape_dim, hidden_dim[1], dim=3)

        kv_input_dim = hidden_dim[0] + hidden_dim[1]

        self.hk_func = MLP(kv_input_dim, hidden_dim[0], hidden_dim[0], norm=mlp_norm, act_fn=act_fn)
        self.hq_func = MLP(hidden_dim[0] + hidden_dim[1], hidden_dim[0], hidden_dim[0], norm=mlp_norm, act_fn=act_fn)
        
        # init message layer to combine scalar features and vector features
        message_layers = []
        for i in range(mess_gvp_layer_num):
            if i == 0:
                mess_input_dim = (hidden_dim[0] + r_feat_dim + edge_feat_dim, hidden_dim[1] + 1)
            else:
                mess_input_dim = hidden_dim
            message_layers.append(
                GVP(mess_input_dim, hidden_dim[1], hidden_dim)
            )
        self.message_layer = nn.Sequential(*message_layers)

        #if l_idx == 0:
        #    node_output_layer_list = [GVP((2 * hidden_dim[0], 2 * hidden_dim[1] + 1), hidden_dim[0], hidden_dim)] + \
        #                            [GVP(hidden_dim, hidden_dim[0], hidden_dim) for _ in range(node_gvp_layer_num-2)] + \
        #                            [GVP(hidden_dim, hidden_dim[0], output_dim)]
        #else:
        node_output_layer_list = [GVP((2 * hidden_dim[0] + input_dim[0], 2 * hidden_dim[1] + 1), hidden_dim[0], hidden_dim)] + \
                                    [GVP(hidden_dim, hidden_dim[0], hidden_dim) for _ in range(node_gvp_layer_num-2)] + \
                                    [GVP(hidden_dim, hidden_dim[0], output_dim)]
            
        if output_norm:
            #self.message_norm = GVPLayerNorm((hidden_dim, hidden_dim))
            self.node_norm = GVPLayerNorm(output_dim)

        self.node_output_layer = nn.Sequential(*node_output_layer_list)

    def embed_ligand_shape(self, scalar_feat, vec_feat, ligand_shape, invar_ligand_shape):
        '''
        embed ligand shape around atom into latent embedding
        '''
        N = scalar_feat.size(0)
        vec_feat_norm = norm_no_nan(vec_feat)
        try:
            net_shape = torch.einsum('bmi,bni->bmn', ligand_shape, vec_feat).view(N, -1)
            shape_input = torch.cat([scalar_feat, net_shape, vec_feat_norm, invar_ligand_shape], -1)
        except:
            pdb.set_trace()

        shape_emb = self.shape_emb_layer(shape_input)
        return shape_emb

    def embed_message_att_weight(self, mess_scalar_emb, mess_vec_emb, node_scalar_emb, node_vec_emb, edge_feat, edge_index, r_feat, rel_x):
        mess_vec_emb_norm = norm_no_nan(mess_vec_emb)
        
        src, dst = edge_index
        node_vec_emb_norm = norm_no_nan(node_vec_emb)

        kv_input = torch.cat([mess_scalar_emb, mess_vec_emb_norm], -1) #r_feat, scalar_vec_emb_i, scalar_vec_emb_j], -1)
        
        # compute k
        k = self.hk_func(kv_input)#.view(-1, self.n_heads, self.output_dim // self.n_heads)
        # compute q
        q = self.hq_func(torch.cat([node_scalar_emb, node_vec_emb_norm], -1))

        # compute attention weight
        att_weight = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0)
        
        return att_weight

    def message_passing(self, scalar_emb, vec_emb, x, ligand_emb, edge_feat, edge_index, r_feat, rel_x):
        N = scalar_emb.size(0)
        src, dst = edge_index
        scalar_emb_i = scalar_emb[src]
        vec_emb_i = vec_emb[src]
        
        ## message embedding
        mess_scalar_i_in = torch.cat([scalar_emb_i, r_feat], -1)
        if edge_feat is not None and self.edge_feat_dim > 0:
            mess_scalar_i_in = torch.cat([mess_scalar_i_in, edge_feat], -1)

        mess_vec_i_in = torch.cat([vec_emb_i, rel_x.unsqueeze(1)], -2)
        #mess_scalar_j_in = scalar_emb_j
        #mess_vec_j_in = vec_emb_j

        mess_scalar_i_out, mess_vec_i_out = self.message_layer((mess_scalar_i_in, mess_vec_i_in))

        ## attention weight
        att_weight = self.embed_message_att_weight(mess_scalar_i_out, mess_vec_i_out, scalar_emb, vec_emb, edge_feat, edge_index, r_feat, rel_x)
        
        ## aggregate messages into atom embeddings
        w_mess_scalar = att_weight.unsqueeze(-1) * mess_scalar_i_out
        w_mess_vec = att_weight.view(-1, 1, 1) * mess_vec_i_out
        scalar_output = scatter_sum(w_mess_scalar, dst, dim=0, dim_size=N)  # (N, heads, H_per_head)
        vec_output = scatter_sum(w_mess_vec, dst, dim=0, dim_size=N)  # (N, heads, H_per_head)
        
        scalar_output = torch.cat([ligand_emb, scalar_emb, scalar_output], -1)

        vec_output = torch.cat([x, vec_emb, vec_output], -2)

        scalar_output, vec_output = self.node_output_layer((scalar_output, vec_output))
        return scalar_output, vec_output

    def forward(self, scalar_feat, vec_feat, r_feat, rel_x, x, ligand_emb, edge_feat, edge_index, ligand_shape, invar_ligand_shape):
        o = self.embed_ligand_shape(scalar_feat, vec_feat, ligand_shape, invar_ligand_shape)

        # update scalar feat with shape embedding o
        scalar_emb = self.shape_scalar_layer(torch.cat([scalar_feat, o], -1))
        
        if self.use_shape_vec_mul:
            vec_o = self.shape_vec_layer(o).unsqueeze(-1).view(scalar_feat.shape[0], -1, self.shape_dim)
            vec_emb_input = torch.cat([vec_feat, vec_o @ ligand_shape], 1)
        else:
            vec_emb_input = torch.cat([vec_feat, ligand_shape], 1)

        vec_emb = self.vector_emb_shape_layer(vec_emb_input)
        
        # message passing
        scalar_output, vec_output = self.message_passing(scalar_emb, vec_emb, x, ligand_emb, edge_feat, edge_index, r_feat, rel_x)
        if self.use_residue and self.output_dim[0] == self.input_dim[0] and self.output_dim[1] == self.input_dim[1]:
            scalar_output = scalar_feat + scalar_output
            vec_output = vec_feat + vec_output
        
        if self.output_norm:
            scalar_output, vec_output = self.node_norm((scalar_output, vec_output))
        return scalar_output, vec_output

class EquivariantShapeEmbLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_layer = VNLinearLeakyReLU(input_dim, output_dim, dim=4)

    def forward(self, shape_h):
        batch_size = shape_h.size(0)
        equiv_shape_h = self.hidden_layer(shape_h)
        return equiv_shape_h

class InvariantShapeEmbLayer(nn.Module):
    def __init__(self, input_dim, output_dim, act_fn='relu', norm=True):
        super().__init__()
        self.hidden_layer = MLP(input_dim, output_dim, input_dim, norm=norm, act_fn=act_fn)

    def forward(self, shape_h):
        batch_size = shape_h.size(0)
        shape_mean = shape_h.mean(dim=1)
        shape_mean_norm = (shape_mean * shape_mean).sum(-1, keepdim=True)
        shape_mean_norm = shape_mean / (shape_mean_norm + EPS)
        
        invar_shape_emb = torch.einsum('bij,bj->bi', shape_h, shape_mean_norm)
        invar_shape_emb = self.hidden_layer(invar_shape_emb)
        return invar_shape_emb
    

class AttentionLayerO2TwoUpdateNodeGeneral(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, num_r_gaussian, edge_feat_dim, shape_dim, act_fn='relu', norm=True,
                 num_gvp=1, r_min=0., r_max=10., num_node_types=8, output_norm=True, use_shape_vec_mul=True, use_residue=True,
                 r_feat_mode='basic', x2h_out_fc=True, sync_twoup=False, pred_bond_type=False, l_idx=0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.num_r_gaussian = num_r_gaussian
        self.norm = norm
        self.act_fn = act_fn
        self.num_gvp = num_gvp
        # self.r2_min = r_min ** 2 if r_min >= 0 else -(r_min ** 2)
        # self.r2_max = r_max ** 2
        self.r_min, self.r_max = r_min, r_max
        self.num_node_types = num_node_types
        self.r_feat_mode = r_feat_mode  # ['origin', 'basic', 'sparse']
        self.pred_bond_type = pred_bond_type

        self.l_idx = l_idx
        self.x2h_out_fc = x2h_out_fc
        self.sync_twoup = sync_twoup
        self.shape_dim = shape_dim
        self.distance_expansion = GaussianSmearing(self.r_min, self.r_max, num_gaussians=num_r_gaussian)
        
        self.gvp_layer = BaseGVPAttLayer(input_dim, hidden_dim, output_dim, shape_dim, n_heads, edge_feat_dim,
                                r_feat_dim=num_r_gaussian * max(edge_feat_dim, 1), use_shape_vec_mul=use_shape_vec_mul,
                                use_residue=use_residue, act_fn=act_fn, mlp_norm=norm, output_norm=output_norm, l_idx=self.l_idx)

        if self.pred_bond_type:
            if self.output_dim[1] == 1:
                self.b_inference = nn.Sequential(
                    nn.Linear(2 * self.output_dim[0] + num_r_gaussian, self.hidden_dim[0]),
                    ShiftedSoftplus(),
                    nn.Linear(self.hidden_dim[0], len(utils_data.BOND_TYPES)),
                )
            else:
                self.b_inference = nn.Sequential(
                    nn.Linear(2 * self.output_dim[0] + 2 * self.output_dim[1], self.hidden_dim[0]),
                    ShiftedSoftplus(),
                    nn.Linear(self.hidden_dim[0], len(utils_data.BOND_TYPES)),
                )

    def _pred_edge_type(self, h, vec, x, edge_index, ligand_bond_index, ligand_bond_type, if_test=False):
        if self.pred_bond_type and not if_test:
            edge_mask, bond_mask = overlap_between_two_tensors(edge_index, ligand_bond_index, dim=1)

        swap_edge_mask = torch.where(edge_index[0, :] >= edge_index[1, :])[0]
        swap_edge_mask_index = torch.index_select(edge_index[:, swap_edge_mask], 0, torch.LongTensor([1,0]).to(edge_index.device))
        unique_edge_index = torch.clone(edge_index)
        unique_edge_index[:, swap_edge_mask] = swap_edge_mask_index

        unique_pred_edge_index, unique_indices = torch.unique(unique_edge_index, sorted=True, return_inverse=True, dim=1)
        
        if self.pred_bond_type and not if_test:
            gt_edge_type = torch.zeros((edge_index.shape[1]), dtype=int).to(ligand_bond_type)
            gt_edge_type[edge_mask] = ligand_bond_type[bond_mask]
            bond_gt = torch.zeros((unique_pred_edge_index.shape[1]), dtype=int).to(ligand_bond_type)
            bond_gt[unique_indices] = gt_edge_type
        else:
            bond_gt = None

        if self.output_dim[1] == 1:
            # if the layer is the last layer with vec equal to the position difference
            src, dst = unique_pred_edge_index
            pred_new_x = vec.squeeze(1) + x
            dist = torch.norm(pred_new_x[dst] - pred_new_x[src], p=2, dim=-1, keepdim=True)
            dist_feat = self.distance_expansion(dist)
            diff_h = torch.abs(h[src] - h[dst])
            sum_h = h[src] + h[dst]
            bond_input = torch.cat((dist_feat, diff_h, sum_h), dim=1)
            bond_pred = self.b_inference(bond_input)
        else:
            src, dst = unique_pred_edge_index
            norm1, norm2 = norm_no_nan(vec[src]), norm_no_nan(vec[dst])
            sum_vec_norm = norm1 + norm2
            diff_vec_norm = torch.abs(norm1 - norm2)
            diff_h = torch.abs(h[src] - h[dst])
            sum_h = h[src] + h[dst]
            bond_input = torch.cat((diff_vec_norm, sum_vec_norm, diff_h, sum_h), dim=1)
            bond_pred = self.b_inference(bond_input)
        
        next_pred_bond_feat = torch.zeros((edge_index.shape[1]), dtype=int).to(ligand_bond_type)
        next_pred_bond_feat = bond_pred[unique_indices]
        return bond_pred, bond_gt, unique_pred_edge_index, next_pred_bond_feat

    def forward(self, h, vec, x, ligand_emb, edge_attr, edge_index, ligand_bond_index, ligand_bond_type, \
                ligand_shape_emb, invar_ligand_shape, if_test=False, l_idx=0):
        if self.edge_feat_dim > 0:
            edge_feat = edge_attr  # shape: [#edges_in_batch, #bond_types]
        else:
            edge_feat = None
        
        src, dst = edge_index
        rel_x = x[dst] - x[src]
        # dist = torch.sum(rel_x ** 2, -1, keepdim=True)
        dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)

        h_in = h
        
        dist_feat = self.distance_expansion(dist)
        dist_feat = outer_product(edge_attr, dist_feat)
        
        new_h, new_vec = self.gvp_layer(h_in, vec, dist_feat, rel_x, x.unsqueeze(1), 
                                ligand_emb, edge_feat, edge_index, ligand_shape_emb, 
                                invar_ligand_shape)
        
        if self.pred_bond_type:
            bond_pred, bond_gt, bond_index, next_bond_feat = self._pred_edge_type(new_h, new_vec, x, edge_index, ligand_bond_index, ligand_bond_type, if_test=if_test)
            return new_h, new_vec, bond_pred, bond_gt, bond_index, next_bond_feat
        else:
            return new_h, new_vec


class UniTransformerO2TwoUpdateGeneral(nn.Module):
    def __init__(self, num_blocks, num_layers, scalar_hidden_dim, vec_hidden_dim, shape_dim, shape_latent_dim, n_heads=1, k=32,
                 num_r_gaussian=50, edge_feat_dim=4, num_node_types=8, act_fn='relu', norm=True, #use_shape=False,
                 cutoff_mode='radius', shape_coeff=0.25, gvp_layer_num=3, r_feat_mode='basic', shape_mode=None, r_max=10., 
                 x2h_out_fc=True, atom_enc_mode='add_aromatic', shape_type='pointAE_shape', sync_twoup=False,
                 pred_bond_type=False, use_shape_vec_mul=True, use_residue=True, time_emb_dim=0):
        super().__init__()
        # Build the network
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.scalar_hidden_dim = scalar_hidden_dim
        self.vec_hidden_dim = vec_hidden_dim
        self.n_heads = n_heads
        self.gvp_layer_num = gvp_layer_num
        self.num_r_gaussian = num_r_gaussian
        self.edge_feat_dim = edge_feat_dim
        self.act_fn = act_fn
        self.norm = norm
        self.num_node_types = num_node_types
        # radius graph / knn graph
        self.cutoff_mode = cutoff_mode  # [radius, none]
        self.k = k
        self.r_feat_mode = r_feat_mode  # [basic, sparse]
        self.atom_enc_mode = atom_enc_mode
        self.shape_type = shape_type
        self.shape_mode = shape_mode
        self.r_max = r_max
        self.x2h_out_fc = x2h_out_fc
        self.sync_twoup = sync_twoup
        self.use_shape_vec_mul = use_shape_vec_mul
        self.use_residue = use_residue
        self.distance_expansion = GaussianSmearing(0., r_max, num_gaussians=num_r_gaussian)
        self.pred_bond_type = pred_bond_type
        self.shape_dim = shape_dim
        self.shape_latent_dim = shape_latent_dim
        self.shape_coeff = shape_coeff
        self.time_emb_dim = time_emb_dim
        self.base_block = self._build_share_blocks()
        if self.edge_feat_dim > 0:
            self.bond_tensors = construct_bond_tensors(self.atom_enc_mode)
            self.bond_margins = (10, 5, 3, 5)
        #if self.shape_mode is not None:
        self.invariant_shape_layer = InvariantShapeEmbLayer(shape_dim, shape_latent_dim)
        self.loss_bond_type = nn.CrossEntropyLoss(reduce=False)
        #self.equivariant_shape_layer = EquivariantShapeEmbLayer(shape_dim, shape_latent_dim // 3)

    def __repr__(self):
        return f'UniTransformerO2(num_blocks={self.num_blocks}, num_layers={self.num_layers}, n_heads={self.n_heads}, ' \
               f'act_fn={self.act_fn}, norm={self.norm}, cutoff_mode={self.cutoff_mode}, ' \
               f'r_feat_mode={self.r_feat_mode}, \n' \
               f'init h emb: {self.init_h_emb_layer.__repr__()} \n' \
               f'base block: {self.base_block.__repr__()} \n' \
               f'edge pred layer: {self.edge_pred_layer.__repr__() if hasattr(self, "edge_pred_layer") else "None"}) '

    def _build_share_blocks(self):
        # Equivariant layers
        base_block = []
        hidden_dims = (self.scalar_hidden_dim, self.vec_hidden_dim)
        for l_idx in range(self.num_layers-1):
            if l_idx == 0:
                input_dims = (self.scalar_hidden_dim, 1)
            else:
                input_dims = hidden_dims
            layer = AttentionLayerO2TwoUpdateNodeGeneral(
                input_dims, hidden_dims, hidden_dims, self.n_heads, self.num_r_gaussian, self.edge_feat_dim, self.shape_dim, 
                act_fn=self.act_fn, norm=self.norm, r_max=self.r_max, num_node_types=self.num_node_types, use_shape_vec_mul=self.use_shape_vec_mul,
                use_residue=self.use_residue, r_feat_mode=self.r_feat_mode, x2h_out_fc=self.x2h_out_fc, output_norm=True, sync_twoup=self.sync_twoup,
                pred_bond_type=self.pred_bond_type, l_idx=l_idx, 
            )
            base_block.append(layer)
        # last layer
        layer = AttentionLayerO2TwoUpdateNodeGeneral(
            hidden_dims, hidden_dims, (self.scalar_hidden_dim, 1), self.n_heads, self.num_r_gaussian, self.edge_feat_dim, self.shape_dim, 
            act_fn=self.act_fn, norm=self.norm, r_max=self.r_max, num_node_types=self.num_node_types, use_shape_vec_mul=self.use_shape_vec_mul,
            use_residue=self.use_residue, r_feat_mode=self.r_feat_mode, x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup, output_norm=False,
            pred_bond_type=self.pred_bond_type, l_idx=self.num_layers,
        )
        base_block.append(layer)
        return nn.ModuleList(base_block)

    def _build_edge_type(self, ligand_v, ligand_pos, edge_index):
        src, dst = edge_index
        atom1_type, atom2_type = torch.argmax(ligand_v[src], dim=1), torch.argmax(ligand_v[dst], dim=1)
        
        atom1_pos, atom2_pos = ligand_pos[src], ligand_pos[dst]
        edge_dist = torch.sqrt(torch.sum((atom1_pos - atom2_pos) ** 2, dim=1))
        edge_type = self._get_bond_order(atom1_type, atom2_type, edge_dist)
        edge_type = F.one_hot(edge_type, num_classes=5)
        return edge_type

    def _get_bond_order(self, atom1, atom2, edge_dist):
        edge_dist = edge_dist * 100
        indices = torch.stack((atom1, atom2), dim=1).to(atom1.device)
        
        edge_types = torch.zeros(atom1.shape[0], dtype=torch.long).to(atom1.device)
        single_bond_indices = torch.where((edge_dist - self.bond_tensors[0][indices[:, 0], indices[:, 1]]) < self.bond_margins[0])[0]
        double_bond_indices = torch.where((edge_dist - self.bond_tensors[1][indices[:, 0], indices[:, 1]]) < self.bond_margins[1])[0]
        triple_bond_indices = torch.where((edge_dist - self.bond_tensors[2][indices[:, 0], indices[:, 1]]) < self.bond_margins[2])[0]
        aromatic_bond_indices = torch.where((edge_dist - self.bond_tensors[3][indices[:, 0], indices[:, 1]]) < self.bond_margins[3])[0]
        
        edge_types[single_bond_indices] = 1
        edge_types[double_bond_indices] = 2
        edge_types[triple_bond_indices] = 3
        edge_types[aromatic_bond_indices] = 4

        return edge_types

    def _find_covalent_indices(self, edge_index, covalent_edge_index):
        tensor_edge_index = edge_index.transpose(1, 0)
        covalent_edge_index = covalent_edge_index.transpose(1, 0)
        
        _, idx, counts = torch.cat([tensor_edge_index, covalent_edge_index], dim=0).unique(dim=0, return_inverse=True, return_counts=True)
        mask = torch.isin(idx, torch.where(counts.gt(1))[0])
        mask1 = mask[:tensor_edge_index.shape[0]]
        covalent_indices = torch.arange(len(mask1))[mask1]
        return covalent_indices

    def _connect_graph(self, ligand_pos, ligand_v, batch):
        edge_index = self._connect_edge(ligand_pos, ligand_v, batch)
        
        if self.cutoff_mode == 'cov_radius':
            # for cov_radius option, both knn graph and covalent radius graph need to be constructed and learned.
            edge_type = self._build_edge_type(edge_index, covalent_index=None)
            covalent_edge_index = self._connect_edge(ligand_pos, ligand_v, batch, cutoff_mode='cov_radius')
            covalent_edge_type = self._build_edge_type(covalent_edge_index, covalent_index=None)
            edge_index = (edge_index, covalent_edge_index)
            edge_type = (edge_type, covalent_edge_type)
        else:
            edge_type = self._build_edge_type(ligand_v, ligand_pos, edge_index)
        return edge_index, edge_type

    def _connect_edge(self, ligand_pos, ligand_v, batch, cutoff_mode='knn'):
        if cutoff_mode == 'knn':
            edge_index = knn_graph(ligand_pos, k=self.k, batch=batch, flow='source_to_target')
        elif cutoff_mode == 'cov_radius':
            edge_index = connect_covalent_graph(ligand_pos, ligand_v, atom_mode=self.atom_enc_mode)
        else:
            raise ValueError(f'Not supported cutoff mode: {self.cutoff_mode}')
        return edge_index

    def _pred_ew(self, x, edge_index):
        src, dst = edge_index
        dist = torch.norm(x[dst] - x[src], p=2, dim=-1, keepdim=True)
        dist_feat = self.distance_expansion(dist)
        logits = self.edge_pred_layer(dist_feat)
        e_w = torch.sigmoid(logits)
        return e_w

    def forward(self, v, h, x, batch_ligand, ligand_shape=None, pred_bond=False, ligand_bond_index=None, ligand_bond_type=None, if_test=False, return_all=None):

        # todo: not sure whether the init embedding layer would be helpful
        # full_src, full_dst = edge_index
        # h, _ = self.init_h_emb_layer(h, x)

        all_vec = [x]
        all_h = [h]
        all_bond_loss = []

        batch_size = torch.max(batch_ligand).item() + 1
        if ligand_shape is not None:
            invar_ligand_shape_emb = self.invariant_shape_layer(ligand_shape)
            invar_ligand_shape_emb = torch.index_select(invar_ligand_shape_emb, 0, batch_ligand)
            
            ligand_shape_emb = torch.index_select(ligand_shape, 0, batch_ligand)
        else:
            ligand_shape_emb = invar_ligand_shape_emb = None

        vec = x.unsqueeze(1)
        ligand_emb = h
        bond_pred = None
        for b_idx in range(self.num_blocks):
            edge_index, edge_type = self._connect_graph(x, v, batch_ligand)
            for l_idx, layer in enumerate(self.base_block):
                if self.pred_bond_type:
                    h, vec, bond_pred, bond_gt, bond_index, next_edge_type = layer(h, vec, x, ligand_emb, edge_type, edge_index, ligand_bond_index, \
                                   ligand_bond_type, ligand_shape_emb, invar_ligand_shape_emb, if_test=if_test)
                    edge_type = F.softmax(next_edge_type, dim=-1)
                    if not if_test:
                        try:
                            batch_edge = torch.index_select(batch_ligand, 0, bond_index[0])
                            loss_level_bond = scatter_mean(self.loss_bond_type(bond_pred, bond_gt), batch_edge, dim=0)
                        except:
                            pdb.set_trace()
                        #if torch.isnan(loss_level_bond).any(): pdb.set_trace()
                        all_bond_loss.append(loss_level_bond)
                else:
                    h, vec = layer(h, vec, x, ligand_emb, edge_type, edge_index, ligand_bond_index, \
                                   ligand_bond_type, ligand_shape_emb, invar_ligand_shape_emb, if_test=if_test)
                    bond_gt = None
                #ligand_emb = h

            #print(f'connect edge time: {t2 - t1}, edge type compute time: {t3 - t2}, forward time: {t4 - t3}')
            vec = vec.squeeze(1)
            all_vec.append(vec)
            all_h.append(h)
        
        pred_new_x = vec + x
        outputs = {'x': pred_new_x, 'h': h, 'bond_pred': bond_pred, 'bond_gt': bond_gt, \
                   'edge_index': bond_index, 'all_bond_loss': all_bond_loss}
        
        if return_all:
            outputs.update({'all_vec': all_vec, 'all_h': all_h})
        return outputs

