import numpy as np
import pandas as pd


from config import get_opt

def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        return lines


def instance_process(lines, maxlen):
    instance_dic = {}
    cnt = 0
    find = False
    word_list_total = []
    for line in lines:
        if line[0:3] == '# i':
            word_list = []
            lemma_list = []
            pos_list = []
            target_idx = [-1, -1]
            span_start = []
            span_end = []
            span_type = []
            length = 0

        elif line[0:3] == '# e':
            instance_dic.setdefault((sent_id, target_type, cnt), {})
            instance_dic[(sent_id, target_type, cnt)]['word_list'] = padding_sentence(word_list, maxlen)
            instance_dic[(sent_id, target_type, cnt)]['lemma_list'] = padding_sentence(lemma_list, maxlen)
            instance_dic[(sent_id, target_type, cnt)]['pos_list'] = padding_sentence(pos_list, maxlen)
            instance_dic[(sent_id, target_type, cnt)]['sent_id'] = sent_id

            word_list_total.append(word_list)
            instance_dic[(sent_id, target_type, cnt)]['length'] = int(length)

            instance_dic[(sent_id, target_type, cnt)]['target_type'] = target_type
            instance_dic[(sent_id, target_type, cnt)]['lu'] = lu
            instance_dic[(sent_id, target_type, cnt)]['target_idx'] = target_idx

            instance_dic[(sent_id, target_type, cnt)]['span_start'] = span_start
            instance_dic[(sent_id, target_type, cnt)]['span_end'] = span_end
            instance_dic[(sent_id, target_type, cnt)]['span_type'] = span_type
            if cnt % 1000 == 0:
                print(cnt)
            cnt += 1
        elif line == '\n':
            continue

        else:
            data_list = line.split('\t')
            word_list.append(data_list[1])
            lemma_list.append(data_list[3])
            pos_list.append(data_list[5])
            sent_id = data_list[6]
            length = data_list[0]

            if data_list[12] != '_' and data_list[13] != '_':
                lu = data_list[12]

                target_type = data_list[13]
                if target_idx == [-1, -1]:
                    target_idx = [int(data_list[0])-1, int(data_list[0])-1]
                else:
                    target_idx[1] =int(data_list[0]) - 1

            if data_list[14] != '_':

                fe = data_list[14].split('-')

                if fe[0] == 'B' and find is False:
                    span_start.append(int(data_list[0]) - 1)
                    find = True

                elif fe[0] == 'O':
                    span_end.append(int(data_list[0]) - 1)
                    span_type.append(fe[-1].replace('\n', ''))
                    find = False

                elif fe[0] == 'S':
                    span_start.append(int(data_list[0]) - 1)
                    span_end.append(int(data_list[0]) - 1)
                    span_type.append(fe[-1].replace('\n', ''))

    return instance_dic

def padding_sentence(sentence: list,maxlen: int):
    while len(sentence) < maxlen:
        sentence.append('<pad>')

    return sentence

class DataConfig:
    def __init__(self,opt):
        self.conll_path = opt.conll_path
        exemplar_lines = load_data(self.conll_path + 'exemplar')
        train_lines = load_data(self.conll_path + 'train')
        dev_lines = load_data(self.conll_path + 'dev')
        test_lines = load_data(self.conll_path + 'test')

        self.emb_file_path = opt.emb_file_path
        self.maxlen = opt.maxlen

        if opt.load_instance_dic:
            self.exemplar_instance_dic = np.load(opt.exemplar_instance_path, allow_pickle=True).item()
            self.train_instance_dic = np.load(opt.train_instance_path, allow_pickle=True).item()
            self.dev_instance_dic = np.load(opt.dev_instance_path, allow_pickle=True).item()
            self.test_instance_dic = np.load(opt.test_instance_path, allow_pickle=True).item()

        else:
            print('begin parsing')
            self.exemplar_instance_dic = instance_process(lines=exemplar_lines,maxlen=self.maxlen)
            np.save(opt.exemplar_instance_path, self.exemplar_instance_dic)
            print('exemplar_instance_dic finish')
            self.train_instance_dic = instance_process(lines=train_lines,maxlen=self.maxlen)
            np.save(opt.train_instance_path, self.train_instance_dic)
            print('train_instance_dic finish')
            self.dev_instance_dic = instance_process(lines=dev_lines,maxlen=self.maxlen)
            np.save(opt.dev_instance_path, self.dev_instance_dic)
            print('dev_instance_dic finish')
            self.test_instance_dic = instance_process(lines=test_lines,maxlen=self.maxlen)
            np.save(opt.test_instance_path, self.test_instance_dic)
            print('test_instance_dic finish')


        self.word_index = {}
        self.word_index['<pad>'] = 0
        self.lemma_index = {}
        self.lemma_index['<pad>'] = 0
        self.pos_index = {}
        self.pos_index['<pad>'] = 0


        self.word_number = 1
        self.lemma_number = 1
        self.pos_number = 1

        self.build_word_index(self.exemplar_instance_dic)
        self.build_word_index(self.train_instance_dic)
        self.build_word_index(self.dev_instance_dic)
        self.build_word_index(self.test_instance_dic)


        self.emb_index = self.build_emb_index(self.emb_file_path)

        self.word_vectors = self.get_embedding_weight(self.emb_index, self.word_index, self.word_number)
        self.lemma_vectors = self.get_embedding_weight(self.emb_index, self.lemma_index, self.lemma_number)

    def build_word_index(self, dic):
        for key in dic:
            word_list =dic[key]['word_list']
            lemma_list = dic[key]['lemma_list']
            pos_list = dic[key]['pos_list']

            for word in word_list:
                if word not in self.word_index:
                    self.word_index[word]=self.word_number
                    self.word_number += 1

            for lemma in lemma_list:
                if lemma not in self.lemma_index:
                    self.lemma_index[lemma]=self.lemma_number
                    self.lemma_number += 1

            for pos in pos_list:
                if pos not in self.pos_index:
                    self.pos_index[pos] = self.pos_number
                    self.pos_number += 1


    def build_emb_index(self, file_path, dim=200):
        with open(file_path, 'r', encoding='utf-8') as data:
            emb_index = {}
            emb_index['<pad>'] = np.zeros(dim, dtype='float32')
            for items in data:
                item = items.split()
                word = item[0]
                weight = np.asarray(item[1:], dtype='float32')
                emb_index[word] = weight

            return emb_index

    def get_embedding_weight(self,embed_dict, words_dict, words_count, dim=200):

        exact_count = 0
        fuzzy_count = 0
        oov_count = 0
        print("loading pre_train embedding by avg for out of vocabulary.")
        embeddings = np.zeros((int(words_count), int(dim)))
        inword_list = np.zeros(int(words_count))
        for word in words_dict:
            if word in embed_dict:
                embeddings[words_dict[word]] = embed_dict[word]
                inword_list[words_dict[word]] = 1
                # 准确匹配
                exact_count += 1
            elif word.lower() in embed_dict:
                embeddings[words_dict[word]] = embed_dict[word.lower()]
                inword_list[words_dict[word]] = 1
                # 模糊匹配
                fuzzy_count += 1
            else:
                # 未登录词
                oov_count += 1
                # embeddings[words_dict[word]] = embed_dict['unk']
                # print(word)
        # 对已经找到的词向量平均化
        sum_col = np.sum(embeddings, axis=0) / len(inword_list)  # avg
        sum_col /= np.std(sum_col)
        embeddings = np.expand_dims(inword_list, -1) * embeddings + (1 - np.expand_dims(inword_list, -1)) * sum_col
        # for i in range(words_count):
        #     if i not in inword_list:
        #         embeddings[i] = sum_col

        # embeddings[int(words_count)] = [0] * dim
        print(f"finish loading... {exact_count} {fuzzy_count} {oov_count}")
        final_embed = np.array(embeddings)
        return final_embed


def load_data_pd(dataset_path,file):
    df=pd.read_csv(dataset_path+file, header=0, encoding='utf-8')
    return df


def get_frame_table(path, file):
    data = load_data_pd(path, file)

    frame_id_to_label = {}
    frame_name_to_label = {}
    frame_name_to_id = {}
    data_index = 0
    for idx in range(len(data['ID'])):
        if data['ID'][idx] not in frame_id_to_label:
            frame_id_to_label[data['ID'][idx]] = data_index
            frame_name_to_label[data['Name'][idx]] = data_index
            frame_name_to_id[data['Name'][idx]] = data['ID'][idx]

            data_index += 1

    return frame_id_to_label, frame_name_to_label, frame_name_to_id


def get_fe_table(path, file):
    data = load_data_pd(path, file)

    fe_id_to_label = {}
    fe_name_to_label = {}
    fe_name_to_id = {}
    fe_id_to_type = {}

    data_index = 0
    for idx in range(len(data['ID'])):
        if data['ID'][idx] not in fe_id_to_label:
            fe_id_to_label[data['ID'][idx]] = data_index
            fe_name_to_label[(data['Name'][idx], data['FrameID'][idx])] = data_index
            fe_name_to_id[(data['Name'][idx], data['FrameID'][idx])] = data['ID'][idx]
            fe_id_to_type[data['ID'][idx]] = data['CoreType'][idx]

            data_index += 1

    return fe_id_to_label, fe_name_to_label, fe_name_to_id, fe_id_to_type


def get_fe_list(path, fe_num, fe_table, frame_id_to_label, opt,  file='FE.csv'):
    fe_dt = load_data_pd(path, file)
    fe_mask_list = {}
    fe_list = {}
    max_fe_num = opt.max_fe_num

    print('begin get fe list')
    for idx in range(len(fe_dt['FrameID'])):
        fe_mask_list.setdefault(fe_dt['FrameID'][idx], [0]*(fe_num+1))
        fe_mask_list[fe_dt['FrameID'][idx]][fe_table[fe_dt['ID'][idx]]] = 1
    for frame_id, fe_lst in fe_mask_list.items():
        fe_lst[fe_num] = 1
    for frame_id, fe_lst in fe_mask_list.items():
        frame_label = frame_id_to_label[frame_id]
        fe_list[frame_label] = {}
        cur_fe_num = np.sum(np.array(fe_lst))
        fe_list[frame_label]['fe_num'] = cur_fe_num
        fe_list[frame_label]['fe'] = np.argsort(-np.array(fe_lst), kind='stable')[:cur_fe_num].tolist() + [fe_num] * (max_fe_num - cur_fe_num) 

    return fe_list


def get_lu_list(path, lu_num, fe_num, frame_id_to_label, fe_list, opt, file='LU.csv'):
    lu_dt = load_data_pd(path, file)
    lu_list = {}
    lu_id_to_name = {}
    lu_name_to_id = {}

    for idx in range(len(lu_dt['ID'])):
        lu_name = lu_dt['Name'][idx]
        lu_list.setdefault(lu_name, {})
        lu_list[lu_name].setdefault('lu_mask', [0]*(lu_num+1))
        lu_list[lu_name]['lu_mask'][frame_id_to_label[lu_dt['FrameID'][idx]]] = 1

        lu_id_to_name[lu_dt['ID'][idx]] = lu_name
        lu_name_to_id[(lu_name, lu_dt['FrameID'][idx])] = lu_dt['ID'][idx]

    for lu_name, dic in lu_list.items():
        lu_mask_list = np.array(dic['lu_mask'])
        dic['frame_num'] = int(np.sum(lu_mask_list))
        dic['frame_list'] = np.argsort(-lu_mask_list, kind='stable')[:opt.max_frame_num].tolist()
        dic['fe_list'] = []
        dic['fe_num'] = []
        for i, frame_id in enumerate(dic['frame_list']):
            if i >= dic['frame_num']:
                dic['fe_list'].append([fe_num] * opt.max_fe_num)
                dic['fe_num'].append(0)
                continue
            dic['lu_mask'][frame_id] = i + 1 
            dic['fe_list'].append(fe_list[frame_id]['fe'])
            dic['fe_num'].append(fe_list[frame_id]['fe_num'])


    return lu_list, lu_id_to_name, lu_name_to_id


if __name__ == '__main__':
    opt = get_opt()
    opt.load_instance_dic = False
    config = DataConfig(opt)
    print(config.word_vectors)
    print(config.lemma_number)
    print(config.word_number)
    print(config.pos_number)
