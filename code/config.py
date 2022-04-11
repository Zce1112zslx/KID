import argparse
import os

data_dir = '../data/'

def get_opt():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--data_path', type=str, default=data_dir + 'parsed-v1.5/')
    parser.add_argument('--conll_path', type=str, default=data_dir+'fn1.5/conll/')
    parser.add_argument('--save_graph_path', type=str, default=data_dir + 'graph/')
    parser.add_argument('--fe_dict_path', type=str, default=data_dir+'fe_label_to_dict.npy')
    parser.add_argument('--graph_name', type=str, default='demo')
    parser.add_argument('--emb_file_path', type=str, default=data_dir + 'glove.6B.200d.txt')
    parser.add_argument('--exemplar_instance_path', type=str, default=data_dir + 'exemplar_instance_dic.npy')
    parser.add_argument('--train_instance_path', type=str, default=data_dir + 'train_instance_dic.npy')
    parser.add_argument('--dev_instance_path', type=str, default=data_dir + 'dev_instance_dic.npy')
    parser.add_argument('--test_instance_path', type=str, default=data_dir + 'test_instance_dic.npy')
    parser.add_argument('--load_instance_dic', type=bool, default=True)
    parser.add_argument('--maxlen', type=int, default=256)
    parser.add_argument('--frame_number',type=int, default=1019)
    parser.add_argument('--role_number',type=int, default=9634)
    parser.add_argument('--dict_number', type=int, default=1169)
    parser.add_argument('--fe_padding_num',type=int, default=5)
    parser.add_argument('--max_frame_num', type=int, default=10)
    parser.add_argument('--max_fe_num', type=int, default=33)

    # train
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--save_model_path', type=str, default='../model/demo.bin')
    parser.add_argument('--pretrain_model_path', type=str, default='none')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--cuda', type=str, default="cuda:0")
    parser.add_argument('--lr', type=float, default='0.0001')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--eval', type=str, default='full')
    parser.add_argument('--seed', type=int, default=1116)
    
    # model
    parser.add_argument('--rnn_hidden_size',type=int, default=256)
    parser.add_argument('--encoder_emb_size',type=int, default=200)
    parser.add_argument('--decoder_emb_size',type=int, default=256)
    parser.add_argument('--pos_emb_size',type=int, default=64)
    parser.add_argument('--token_type_emb_size',type=int, default=36)
    parser.add_argument('--cell_name', type=str, default='lstm')
    parser.add_argument('--node_emb_size',type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--decoder_hidden_size',type=int, default=256)
    return parser.parse_args()