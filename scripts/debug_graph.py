import torch
import numpy as np
## Data training: B = 1 (molecular) |||| Nc = 5 -> 5 graphs (same molecular) -> 22 nodes (5,22)
## flattening (5, 22*22) | (5,22,128)
sample_template = torch.load("./share_dir/sample_template.pt")
# list_node_feats = torch.load("./share_dir/list_node_feats.pt") ## [5,22,64] [Nc*B, 22, 64] --> decompose [B, Nc, 22, 64]
# list_adjs = torch.load("./share_dir/list_adjs.pt") ## 
C_bary = torch.load("./share_dir/C_bary.pt") ## [5,22,64]
# print(f"list_adjs: {list_adjs[0]}")
print(f"C_bary: {C_bary.shape}")
# for item in list_adjs:
#     print(torch.equal(C_bary, item))
# print(f"list_node_feats: {list_node_feats[0].shape, list_node_feats[0]}")
# F_bary = torch.load("./share_dir/F_bary.pt")  ## [22,64]
# print(f"F_bary: {F_bary.shape, F_bary}")
# for item in list_node_feats:
#     print(torch.equal(F_bary, item))
## dict_keys(['input', 'edge_index', 'edge_attr', 'batch', 'adjacency', 'output'])
# print(f"input: {sample_template['input'], sample_template['input'].shape}") ## before GCN
# print(f"edge_index: {sample_template['edge_index'], sample_template['edge_index'].shape}")
print(f"h: {sample_template['h'], sample_template['h'].shape}")
print(f"output: {sample_template['output'], sample_template['output'].shape}")
# print(f"batch: {sample_template['batch'], sample_template['batch'].shape}")
# from torch_geometric.utils import to_dense_batch, to_dense_adj
# out_feat, out_mask = to_dense_batch(
#     x=sample_template['h'].detach().cpu(),
#     batch=sample_template['batch'].detach().cpu()
# )

# def get_list_node_features(out, mask):
#     """
#     Convert outputs of to_dense_batch to a list of different feature node matrices
#     :param out:
#     :param mask:
#     :return:
#     """
#     our_filter = []
#     for index, sample in enumerate(out):
#         # print (f"Sample size: {sample.shape}")
#         # print(f"Sample value: {sample}")
#         # print (f"map size : {mask[index].shape}")
#         # print(f"map value : {mask[index]}")
#         mask_index = mask[index]
#         pos_true = (mask_index == True).nonzero().squeeze()
#         # print(pos_true)
#         our_filter.append(sample[pos_true])
#         # print(f"our filrer: {our_filter}")
#     return our_filter

# list_node_feat = get_list_node_features(out_feat, out_mask)
# print(sample_template["edge_attr"].get_device())

# print(sample_template["edge_attr"].to(sample_template["edge_index"].get_device()))

# adj = to_dense_adj(
#     edge_index=sample_template["edge_index"].detach().cpu(),
#     batch=sample_template['batch'].detach().cpu(),
#     edge_attr=sample_template['edge_attr'].detach().cpu()
# )
# list_adj = [item for item in adj]

# print(f"list_adj: {len(list_adj), list_adj[0].shape}")
# # print(sample_template["adjacency"].shape)
# # batch = sample_template["batch"].detach().cpu().numpy()
# # for b in np.unique(batch):
# #     nodes = batch[batch==b]
# #     print(f"conformer {b} has {len(nodes)} nodes")
# # print(sample_template["batch"], sample_template["batch"].shape)