import scipy.sparse as sparse
import numpy as np
import os

from config import get_opt
from data_preprocess import load_data_pd, get_frame_table, get_fe_table

def build_adj(row: list, col: list):
    eps = 1e-12
    row = np.array(row)
    col = np.array(col)

    data = np.ones_like(row)
    A = sparse.coo_matrix((data, (row, col)))
    D = sparse.diags(np.power(np.array(A.sum(1) + eps), -0.5).flatten())
    A_hat_coo = D.dot(A.dot(D)).tocoo().astype(np.float32)
    return A_hat_coo

def build_graph(data_path, graph_path):
    frame_id_to_label, frame_name_to_label, \
    frame_name_to_id = get_frame_table(data_path, 'frame.csv')

    fe_id_to_label, fe_name_to_label, fe_name_to_id, \
    fe_id_to_type = get_fe_table(data_path, 'FE.csv')

    frame_num = len(frame_id_to_label)
    fe_num = len(fe_id_to_label)

    if not os.path.exists(graph_path):
        os.makedirs(graph_path)
    
    adj_list = []

    row = []
    col = []

    # frame - fe
    fe_dt = load_data_pd(data_path, 'FE.csv')
    for idx in range(len(fe_dt['FrameID'])):
        frame_node_idx = frame_id_to_label[fe_dt['FrameID'][idx]]
        fe_node_idx = frame_num + fe_id_to_label[fe_dt['ID'][idx]]
        row.append(frame_node_idx)
        col.append(fe_node_idx)
        row.append(fe_node_idx)
        col.append(frame_node_idx)
    adj = build_adj(row, col)
    adj_list.append(adj)
    sparse.save_npz(graph_path+'frame_fe.npz', adj)

    # inter frame: frame-frame, fe-fe
    row = []
    col = []
    frame_relation_dt = load_data_pd(data_path, 'frameRelations.csv')
    for idx in range(len(frame_relation_dt['subID'])):
        subID = frame_relation_dt['subID'][idx]
        supID = frame_relation_dt['supID'][idx]
        sub_frame_node_idx = frame_id_to_label[subID]
        sup_frame_node_idx = frame_id_to_label[supID]
        row.append(sub_frame_node_idx)
        col.append(sup_frame_node_idx)
        row.append(sup_frame_node_idx)
        col.append(sub_frame_node_idx)
    adj = build_adj(row, col)
    adj_list.append(adj)
    sparse.save_npz(graph_path+'frame_frame.npz', adj)

    row = []
    col = []
    fe_relation_dt = load_data_pd(data_path, 'feRelations.csv')
    for idx in range(len(fe_relation_dt['subID'])):
        sub_fe_node_idx = frame_num + fe_id_to_label[fe_relation_dt['subID'][idx]]
        sup_fe_node_idx = frame_num + fe_id_to_label[fe_relation_dt['supID'][idx]]
        row.append(sup_fe_node_idx)
        col.append(sub_fe_node_idx)
        row.append(sub_fe_node_idx)
        col.append(sup_fe_node_idx)
    adj = build_adj(row, col)
    adj_list.append(adj)
    sparse.save_npz(graph_path+'inter_fe.npz', adj)

    # intra - frame fe-to-fe
    row = []
    col = []
    fe_adj = np.load('../data/intra_frame_fe_relations.npy', allow_pickle=True).item()
    for k, v in fe_adj.items():
        for vv in v:
            row.append(frame_num + k)
            col.append(frame_num + vv)
            row.append(frame_num + vv)
            col.append(frame_num + k)
    adj = build_adj(row, col)
    adj_list.append(adj)
    sparse.save_npz(graph_path+'intra_fe.npz', adj)
    # self-loop

    row = []
    col = []
    for i in range(frame_num + fe_num + 1):
        row.append(i)
        col.append(i)
    adj = build_adj(row, col)
    adj_list.append(adj)
    sparse.save_npz(graph_path+'self_loop.npz', adj)

    # build adj matrix



    # save adj matrix
    # sparse.save_npz(graph_path+graph_name+'.npz', A_hat_coo)
    return adj_list


if __name__ == '__main__':
    opt = get_opt()
    g_list = build_graph(opt.data_path, opt.save_graph_path)
    for g in g_list:
        print(g.shape)
        # print(len(g.data))
        print(len(g.row))
        # print(g.todense().shape)
        

