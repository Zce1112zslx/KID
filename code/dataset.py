import torch 
import numpy as np
import scipy.sparse as sparse
import random

from torch.utils.data import Dataset, DataLoader
from config import get_opt
from data_preprocess import get_frame_table, get_fe_table, get_fe_list, get_lu_list,DataConfig
from utils import get_mask_from_index

class FrameNetDataset(Dataset):
    def __init__(self, opt, config, data_dic, device):
        super(FrameNetDataset, self).__init__()
        print('loading data...')
        # data_dic: keys (sent_id, target_type, cnt)
        self.data = []
        self.data_dic = data_dic
        # id/name: frame id/name in frame.csv | label: label for classification
        self.frame_id_to_label, self.frame_name_to_label, \
        self.frame_name_to_id = get_frame_table(opt.data_path, 'frame.csv')

        # id/name/type: fe id/name/type in FE.csv | label: label for classification
        self.fe_id_to_label, self.fe_name_to_label, self.fe_name_to_id, \
        self.fe_id_to_type = get_fe_table(opt.data_path, 'FE.csv')
        
        self.word_index = config.word_index
        self.lemma_index = config.lemma_index
        self.pos_index = config.pos_index
        # self.rel_index = config.rel_index

        self.fe_num = len(self.fe_id_to_label)
        self.frame_num = len(self.frame_id_to_label)
        self.batch_size = opt.batch_size
        print(f'frame num {self.frame_num} FE num {self.fe_num}')
        # print(self.frame_num)
        self.dataset_len = len(self.data_dic)

        # frame id -> its FEs with 1 and others with 0
        # self.fe_mask_list = get_fe_list(opt.data_path, self.fe_num, self.fe_id_to_label)
        self.fe_list = get_fe_list(opt.data_path, self.fe_num, self.fe_id_to_label, self.frame_id_to_label, opt)
        # lu_list: dic 'fe_mask' and 'lu_mask'
        # lu_id_to_name: lu ids -> name
        # lu_name_to_id: (LU name, frameID) -> LU id 
        self.lu_list, self.lu_id_to_name,\
        self.lu_name_to_id = get_lu_list(opt.data_path,
                                          self.frame_num, self.fe_num,
                                          self.frame_id_to_label,
                                          self.fe_list, 
                                          opt)


        self.device = device
        self.oov_frame = 0
        self.long_span = 0
        self.error_span = 0
        self.error_syntax = 0
        self.fe_coretype_table = {}
        self.target_mask = {}
        self.graph_list_len = opt.fe_padding_num + 1
        self.maxlen = opt.maxlen

        for idx, fe_type in self.fe_id_to_type.items():
            if fe_type == 'Core':
                self.fe_coretype_table[self.fe_id_to_label[idx]] = 1
            else:
                self.fe_coretype_table[self.fe_id_to_label[idx]] = 0



        for key in self.data_dic.keys():
            self.build_target_mask(key,opt.maxlen)


        for key in self.data_dic.keys():
            self.pre_process(key, opt)

        self.pad_dic_cnt = (opt.batch_size - self.dataset_len % opt.batch_size) % opt.batch_size


        for idx,key in enumerate(self.data_dic.keys()):
            if idx >= self.pad_dic_cnt:
                break
            self.pre_process(key, opt,filter=False)

        self.dataset_len+=self.pad_dic_cnt
        
        print('load data finish')
        print('oov frame = ', self.oov_frame)
        print('long_span = ', self.long_span)
        print('error_syntax = ', self.error_syntax)
        print('dataset_len = ', self.dataset_len)

    def __len__(self):
        self.dataset_len = int(self.dataset_len / self.batch_size) * self.batch_size
        return self.dataset_len

    def __getitem__(self, item):
        return self.data[item]

    def pre_process(self, key, opt,filter=True):
        dic = {}
        instance = self.data_dic[key]
        if instance['target_type'] not in self.frame_name_to_label:
            self.oov_frame += 1
            self.dataset_len -= 1
            return
        if len(instance['dep_list'][0]) != self.maxlen:
            self.error_syntax += 1
            self.dataset_len -= 1
            return
        target_id = self.frame_name_to_id[instance['target_type']]
        if filter:
            self.long_span += self.remove_error_span(key, instance['span_start'],
                                                 instance['span_end'], instance['span_type'], target_id, 20)

        word_ids = [self.word_index[word] for word in instance['word_list']]
        dic['word'] = word_ids
        lemma_ids = [self.lemma_index[lemma] for lemma in instance['lemma_list']]
        dic['lemma'] = lemma_ids
        pos_ids = [self.pos_index[pos] for pos in instance['pos_list']]
        dic['pos'] = pos_ids
        dic['length'] = instance['length']


        dic['target_head'] = instance['target_idx'][0]
        dic['target_tail'] = instance['target_idx'][1]

        mask = get_mask_from_index(torch.Tensor([int(instance['length'])]), opt.maxlen).squeeze()
        dic['mask'] = mask

        token_type_ids = build_token_type_ids(instance['target_idx'][0], instance['target_idx'][1], opt.maxlen)
        dic['token_type'] = token_type_ids
        dic['target_mask'] = self.target_mask[key[0]]
        target_label = self.frame_name_to_label[instance['target_type']]
        dic['target_type'] = target_label

        if instance['length'] <= opt.maxlen:
            sent_length = instance['length']
        else:
            sent_length = opt.maxlen
        dic['sent_length'] = sent_length

        lu_name = instance['lu']
        lu_dic = self.lu_list[lu_name]
        dic['lu_mask'] = lu_dic['lu_mask']
        dic['lu_frame_num'] = int(lu_dic['frame_num'])
        
        lu_frame_list = lu_dic['frame_list']
        dic['frame_list'] = lu_frame_list
        gold_frame = -1
        for i in range(opt.max_frame_num):
            if target_label == lu_frame_list[i]:
                gold_frame = i
                dic['gold_frame_id'] = gold_frame
                break
        if gold_frame == -1:
            print("error")
        dic['frame_fe_num'] = lu_dic['fe_num']
        fe_list = lu_dic['fe_list']
        dic['fe_list'] = fe_list


        fe_head = instance['span_start']
        fe_tail = instance['span_end']



        while len(fe_head) < opt.fe_padding_num:
            fe_head.append(min(sent_length-1, opt.maxlen-1))

        while len(fe_tail) < opt.fe_padding_num:
            fe_tail.append(min(sent_length-1,opt.maxlen-1))


        dic['fe_head'] = fe_head[0:opt.fe_padding_num]
        dic['fe_tail'] = fe_tail[0:opt.fe_padding_num]

        fe_type = [self.fe_name_to_label[(item, target_id)] for item in instance['span_type']]


        dic['fe_cnt'] = min(len(fe_type), opt.fe_padding_num)

        while len(fe_type) < opt.fe_padding_num:
            fe_type.append(self.fe_num)

        row_indicies = []
        col_indicies = []
        r = [0]
        c= [0]
        for t in range(self.graph_list_len):
            row_indicies.append([rr for rr in r])
            col_indicies.append([cc for cc in c])
            r += [0, t+1, t+1]
            c += [t+1, 0, t+1]
        dic['row'] = row_indicies
        dic['col'] = col_indicies

        dep_row = []
        dep_col = []
        head_list = instance['dep_list'][0]
        for i, j in enumerate(head_list):
            dep_row.append(i)
            dep_col.append(i)
            if i == j:
                continue
            dep_row.append(i)
            dep_col.append(j)
            dep_row.append(j)
            dep_col.append(i)
        dic['dep_row'] = dep_row
        dic['dep_col'] = dep_col
        dic['fe_type'] = fe_type[0:opt.fe_padding_num]
        self.data.append(dic)

    def remove_error_span(self, key, fe_head_list, fe_tail_list, fe_type_list, target_id, span_maxlen):
        indices = []
        for index in range(len(fe_head_list)):
            if fe_tail_list[index] - fe_head_list[index] >= span_maxlen:
                indices.append(index)
            elif fe_tail_list[index] < fe_head_list[index]:
                indices.append(index)


            elif (fe_type_list[index], target_id) not in self.fe_name_to_label:
                indices.append(index)

            else:
                for i in range(index):
                    if i not in indices:
                        if fe_head_list[index] >= fe_head_list[i] and fe_head_list[index] <= fe_tail_list[i]:
                            indices.append(index)
                            break

                        elif fe_tail_list[index] >= fe_head_list[i] and fe_tail_list[index] <= fe_tail_list[i]:
                            indices.append(index)
                            break
                        elif fe_tail_list[index] <= fe_head_list[i] and fe_tail_list[index] >= fe_tail_list[i]:
                            indices.append(index)
                            break
                        else:
                            continue

        fe_head_list_filter = [i for j, i in enumerate(fe_head_list) if j not in indices]
        fe_tail_list_filter = [i for j, i in enumerate(fe_tail_list) if j not in indices]
        fe_type_list_filter = [i for j, i in enumerate(fe_type_list) if j not in indices]
        self.data_dic[key]['span_start'] = fe_head_list_filter
        self.data_dic[key]['span_end'] = fe_tail_list_filter
        self.data_dic[key]['span_type'] = fe_type_list_filter

        return len(indices)

    def build_target_mask(self,key,maxlen):
        self.target_mask.setdefault(key[0], [0]*maxlen)

        target_head = self.data_dic[key]['target_idx'][0]
        target_tail = self.data_dic[key]['target_idx'][1]
        self.target_mask[key[0]][target_head] = 1
        self.target_mask[key[0]][target_tail] = 1


class FrameNetDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle):
        super(FrameNetDataLoader, self).__init__(dataset, batch_size=batch_size)
        self.shuffle = shuffle
        self.length = len(dataset)
        self.order = list(range(self.length))
        self.device = dataset.device
        self.graph_list_len = dataset.graph_list_len
        self.maxlen = dataset.maxlen

    def gen_A(self, row, col):
        data = np.ones_like(row)
        A = sparse.coo_matrix((data, (row, col)))
        D = sparse.diags(np.power(np.array(A.sum(1)), -0.5).flatten())
        A_hat_coo = D.dot(A.dot(D)).tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((A_hat_coo.row, A_hat_coo.col)).astype(np.int64))
        values = torch.from_numpy(A_hat_coo.data)
        A_hat = torch.sparse_coo_tensor(indices, values, torch.Size(A_hat_coo.shape)).to(self.device)
        return A_hat
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.order)
        self.data = [self.dataset[idx] for idx in self.order]
        batch_num = self.length // self.batch_size
        self.batches = [self.data[idx*self.batch_size:(idx+1)*self.batch_size] for idx in range(batch_num)]
        for idx, mini_batch in enumerate(self.batches):
            word_ids = []
            lemma_ids = []
            pos_ids = []

            lengths = []
            mask = []
            target_head = []
            target_tail = []
            target_type = []
            fe_head = []
            fe_tail = []
            fe_type = []
            sent_length = []
            fe_cnt = []
            lu_mask = []
            lu_frame_num = []
            frame_fe_num = []
            gold_frame_id = []
            frame_list = [] # len = 10
            fe_list = [] # 10 * 33
            token_type_ids = []
            target_mask_ids = []
            row_lists = [[] for i in range(self.graph_list_len)]
            col_lists = [[] for i in range(self.graph_list_len)]
            A_hat_list = []
            dep_row = []
            dep_col = []

            for i, example in enumerate(mini_batch):
                word_ids.append(torch.Tensor(example['word']).long().unsqueeze(0))
                lemma_ids.append(torch.Tensor(example['lemma']).long().unsqueeze(0))
                pos_ids.append(torch.Tensor(example['pos']).long().unsqueeze(0))
                lengths.append(torch.Tensor([example['length']]).long().unsqueeze(0))
                mask.append(example['mask'].long().unsqueeze(0))
                target_head.append(torch.Tensor([example['target_head']]).long().unsqueeze(0))
                target_tail.append(torch.Tensor([example['target_tail']]).long().unsqueeze(0))
                target_type.append(torch.Tensor([example['target_type']]).long().unsqueeze(0))
                fe_head.append(torch.Tensor(example['fe_head']).long().unsqueeze(0))
                fe_tail.append(torch.Tensor(example['fe_tail']).long().unsqueeze(0))
                fe_type.append(torch.Tensor(example['fe_type']).long().unsqueeze(0))
                fe_cnt.append(torch.Tensor([example['fe_cnt']]).long().unsqueeze(0))
                lu_mask.append(torch.Tensor(example['lu_mask']).long().unsqueeze(0))
                gold_frame_id.append(torch.Tensor([example['gold_frame_id']]).long().unsqueeze(0))
                lu_frame_num.append(torch.Tensor([example['lu_frame_num']]).long().unsqueeze(0))
                frame_list.append(torch.Tensor(example['frame_list']).long().unsqueeze(0))
                frame_fe_num.append(torch.Tensor(example['frame_fe_num']).long().unsqueeze(0))
                fe_list.append(torch.Tensor(example['fe_list']).long().unsqueeze(0))
                token_type_ids.append(torch.Tensor(example['token_type']).long().unsqueeze(0))
                sent_length.append(torch.Tensor([example['sent_length']]).long().unsqueeze(0))
                target_mask_ids.append(torch.Tensor(example['target_mask']).long().unsqueeze(0))
                dep_row += [x + self.maxlen*i for x in example['dep_row']]
                dep_col += [x + self.maxlen*i for x in example['dep_col']]
                for j in range(self.graph_list_len):
                    row_lists[j] += [x + (j + 1)*i for x in example['row'][j]]
                    col_lists[j] += [x + (j + 1)*i for x in example['col'][j]]
            for j in range(self.graph_list_len):
                row = np.array(row_lists[j])
                col = np.array(col_lists[j])
                A_hat_list.append(self.gen_A(row, col))
            dep_A_hat = self.gen_A(dep_row, dep_col)
            yield {
                'word': torch.cat(word_ids, dim=0).to(self.device),
                'lemma': torch.cat(lemma_ids, dim=0).to(self.device),
                'pos': torch.cat(pos_ids, dim=0).to(self.device),
                'length': torch.cat(lengths, dim=0).to(self.device),
                'mask': torch.cat(mask, dim=0).to(self.device),
                'target_head': torch.cat(target_head, dim=0).to(self.device),
                'target_tail': torch.cat(target_tail, dim=0).to(self.device),
                'target_type': torch.cat(target_type, dim=0).to(self.device),
                'fe_head': torch.cat(fe_head, dim=0).to(self.device),
                'fe_tail': torch.cat(fe_tail, dim=0).to(self.device),
                'fe_type': torch.cat(fe_type, dim=0).to(self.device),
                'fe_cnt': torch.cat(fe_cnt, dim=0).to(self.device),
                'lu_mask': torch.cat(lu_mask, dim=0).to(self.device),
                'gold_frame_id': torch.cat(gold_frame_id, dim=0).to(self.device),
                'lu_frame_num': torch.cat(lu_frame_num, dim=0).to(self.device),
                'frame_list': torch.cat(frame_list, dim=0).to(self.device),
                'frame_fe_num': torch.cat(frame_fe_num, dim=0).to(self.device),
                'fe_list': torch.cat(fe_list, dim=0).to(self.device),
                'token_type': torch.cat(token_type_ids, dim=0).to(self.device),
                'sent_length': torch.cat(sent_length, dim=0).to(self.device),
                'target_mask': torch.cat(target_mask_ids, dim=0).to(self.device),
                'adj_list': A_hat_list,
                'dep': dep_A_hat,
            }

def build_token_type_ids(target_head, target_tail, maxlen):
    token_type_ids = [0]*maxlen
    token_type_ids[target_head] = 1
    token_type_ids[target_tail] = 1

    return token_type_ids


if __name__ == '__main__':
    opt = get_opt()
    config = DataConfig(opt)
    if torch.cuda.is_available():
        device = torch.device(opt.cuda)
    else:
        device = torch.device('cpu')
    print(device)
    dataset = FrameNetDataset(opt, config, config.dev_instance_dic, device)
    print(dataset.error_span)
    print("building loader")
    dl = FrameNetDataLoader(
                dataset,
                batch_size=opt.batch_size,
                shuffle=True
            )
    print("loader built")
    for b in dl:
        print(b['adj_list'])
        break