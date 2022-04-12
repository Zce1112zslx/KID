from data_preprocess import load_data_pd, get_frame_table, get_fe_table
from config import get_opt
import numpy as np


def build_fe_dict(data_path):
    frame_id_to_label, frame_name_to_label, \
    frame_name_to_id = get_frame_table(data_path, 'frame.csv')

    fe_id_to_label, fe_name_to_label, fe_name_to_id, \
    fe_id_to_type = get_fe_table(data_path, 'FE.csv')

    fe_dt = load_data_pd(data_path, 'FE.csv')

    name_dict = {}
    name_cnt = 0
    fe_label_to_dict = {}

    for idx in range(len(fe_dt['Name'])):
        name = fe_dt['Name'][idx]
        fe_id = fe_dt['ID'][idx]
        fe_label = fe_id_to_label[fe_id]
        if name not in name_dict:
            name_dict[name] = name_cnt
            name_cnt += 1
        fe_label_to_dict[fe_label] = name_dict[name]
    
    fe_label_to_dict[len(fe_dt['Name'])] = name_cnt

    return fe_label_to_dict, name_dict

def build_fe_dict2(data_path):
    frame_id_to_label, frame_name_to_label, \
    frame_name_to_id = get_frame_table(data_path, 'frame.csv')

    fe_id_to_label, fe_name_to_label, fe_name_to_id, \
    fe_id_to_type = get_fe_table(data_path, 'FE.csv')

    fe_dt = load_data_pd(data_path, 'FE.csv')

    name_dict = {}
    name_cnt = 0
    fe_label_to_dict = {}

    for idx in range(len(fe_dt['Name'])):
        name = fe_dt['Name'][idx]
        fe_id = fe_dt['ID'][idx]
        fe_label = fe_id_to_label[fe_id]
        fe_type = fe_id_to_type[fe_id]
        if fe_type == 'Core':
            fe_label_to_dict[fe_label] = name_cnt
            name_cnt += 1
            continue
        if name not in name_dict:
            name_dict[name] = name_cnt
            name_cnt += 1
        fe_label_to_dict[fe_label] = name_dict[name]
    
    fe_label_to_dict[len(fe_dt['Name'])] = name_cnt

    return fe_label_to_dict, name_dict, name_cnt

if __name__ == '__main__':
    opt = get_opt()
    dic, _ = build_fe_dict(opt.data_path)
    print(dic)
    print(_)
    # print(cnt)
    # np.save('../data/fe_label_to_dict2.npy', dic, allow_pickle=True)
    # print(len(dic))