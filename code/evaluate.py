import torch
import numpy as np

class Eval():
    def __init__(self,opt,Dataset):
        self.fe_TP = 0
        self.fe_TP_FP = 0
        self.fe_TP_FN = 0

        self.frame_cnt = 0
        self.frame_acc = 0

        self.core_cnt = 0
        self.nocore_cnt = 0

        self.decay = 0.8
        self.opt=opt

        self.fe_id_to_label =Dataset.fe_id_to_label
        self.fe_label_to_id = {}
        for key, val in self.fe_id_to_label.items():
            self.fe_label_to_id[val] = key

        self.frame_id_to_label =Dataset.frame_id_to_label
        self.frame_label_to_id = {}
        for key, val in self.frame_id_to_label.items():
            self.frame_label_to_id[val] = key

        path = '../data/frame-fe-dist-path/'
        self.fe_dis_matrix = np.load(path + 'fe_dis_matrix.npy', allow_pickle=True)
        self.frame_dis_matrix = np.load(path + 'frame_dis_matrix.npy', allow_pickle=True)
        self.fe_path_matrix = np.load(path + 'fe_path_matrix.npy', allow_pickle=True)
        self.frame_path_matrix = np.load(path + 'frame_path_matrix.npy', allow_pickle=True)
        self.fe_hash_index = np.load(path + 'fe_hash_idx.npy', allow_pickle=True).item()
        self.frame_hash_index = np.load(path + 'frame_hash_idx.npy', allow_pickle=True).item()

    def metrics(self,batch_size:int,fe_cnt:torch.Tensor,\
                gold_fe_type:torch.Tensor,gold_fe_head:torch.Tensor,\
                gold_fe_tail:torch.Tensor,gold_frame_type:torch.Tensor,\
                pred_fe_type:torch.Tensor,pred_fe_head:torch.Tensor,\
                pred_fe_tail:torch.Tensor,pred_frame_type:torch.Tensor,
                fe_coretype,sent_length:torch.Tensor):

        self.frame_cnt+=batch_size
        for batch_index in range(batch_size):
            #caculate frame acc
            # print(gold_frame_type[batch_index])
            # print(pred_frame_type[batch_index])
            if self.frame_label_to_id[int(gold_frame_type[batch_index])] in self.frame_hash_index.keys() and \
                self.frame_label_to_id[int(pred_frame_type[0][batch_index])] in self.frame_hash_index.keys():
                gold_frame_idx = self.frame_hash_index[self.frame_label_to_id[int(gold_frame_type[batch_index])]]
                pred_frame_idx = self.frame_hash_index[self.frame_label_to_id[int(pred_frame_type[0][batch_index])]]
                KeyError = False
            else:
                KeyError = True

            if gold_frame_type[batch_index] == pred_frame_type[0][batch_index]:
                self.frame_acc += 1

            elif KeyError is False:
                if self.frame_dis_matrix[gold_frame_idx][pred_frame_idx]!=-1 and self.frame_dis_matrix[gold_frame_idx][pred_frame_idx] <=9:
                    self.frame_acc += 0.8**self.frame_dis_matrix[gold_frame_idx][pred_frame_idx]
            # update tp_fn
            # self.fe_TP_FN+=int(fe_cnt[batch_index])
            for fe_index in range(fe_cnt[batch_index]):
                if fe_coretype[int(gold_fe_type[batch_index][fe_index])] == 1:
                    self.fe_TP_FN+=1
                    self.core_cnt+=1
                else:
                    self.fe_TP_FN+=0.5
                    self.nocore_cnt+=1

            # preprocess error

            #gold_fe_list = gold_fe_type.cpu().numpy().tolist()
            gold_tail_list =gold_fe_tail.cpu().numpy().tolist()
            #update fe_tp and fe_TP_FP
            for fe_index in range(self.opt.fe_padding_num):
                # if pred_fe_tail[fe_index][batch_index] == sent_length[batch_index]-1 and \
                #    pred_fe_head[fe_index][batch_index] == sent_length[batch_index] - 1:
                if pred_fe_type[fe_index][batch_index] == self.opt.role_number:
                    break


                #update fe_tp
                if pred_fe_tail[fe_index][batch_index] in gold_tail_list[batch_index]:
                    idx = gold_tail_list[batch_index].index(pred_fe_tail[fe_index][batch_index])



                    if pred_fe_head[fe_index][batch_index]==gold_fe_head[batch_index][idx] and \
                            pred_fe_type[fe_index][batch_index] == gold_fe_type[batch_index][idx] :
                        if fe_coretype[int(gold_fe_type[batch_index][idx])] == 1:
                            self.fe_TP+= 1
                        else:
                            self.fe_TP+= 0.5
                    
                    
                    elif gold_fe_type[batch_index][idx]==self.opt.role_number:
                        pass

                    elif KeyError is True:
                        pass

                    elif pred_fe_head[fe_index][batch_index]==gold_fe_head[batch_index][idx] and \
                        self.fe_label_to_id[int(gold_fe_type[batch_index][idx])] in self.fe_hash_index.keys() \
                            and self.fe_label_to_id[int(pred_fe_type[fe_index][batch_index])] in self.fe_hash_index.keys():
                        gold_fe_idx = self.fe_hash_index[self.fe_label_to_id[int(gold_fe_type[batch_index][idx])]]
                        pred_fe_idx = self.fe_hash_index[self.fe_label_to_id[int(pred_fe_type[fe_index][batch_index])]]
                        
                        if self.fe_dis_matrix[gold_fe_idx][pred_fe_idx] != -1 and self.fe_dis_matrix[gold_fe_idx][pred_fe_idx] < 9 :
                            rate = float(self.fe_path_matrix[gold_fe_idx][pred_fe_idx]/(self.frame_path_matrix[gold_frame_idx][pred_frame_idx]+0.000001))
                            if rate > 1:
                                rate = 1
                            if fe_coretype[int(gold_fe_type[batch_index][idx])] == 1:
                                self.fe_TP += 1*rate*(0.8**self.fe_dis_matrix[gold_fe_idx][pred_fe_idx])
                            else:
                                self.fe_TP += 0.5*rate*(0.8**self.fe_dis_matrix[gold_fe_idx][pred_fe_idx])

                #update fe_tp_fp
                if fe_coretype[int(pred_fe_type[fe_index][batch_index])] == 1:
                    self.fe_TP_FP += 1
                else:
                    self.fe_TP_FP += 0.5

    def calculate(self):
        frame_acc = self.frame_acc / self.frame_cnt
        fe_prec = self.fe_TP / (self.fe_TP_FP+0.000001)
        fe_recall = float(self.fe_TP / self.fe_TP_FN)
        fe_f1 = 2*fe_prec*fe_recall/(fe_prec+fe_recall+0.0000001)

        full_TP = self.frame_acc+self.fe_TP
        full_TP_FP = self.frame_cnt + self.fe_TP_FP
        full_TP_FN = self.frame_cnt +self.fe_TP_FN

        full_prec = float(full_TP / full_TP_FP)
        full_recall =float(full_TP / full_TP_FN)
        full_f1 = 2 * full_prec * full_recall / (full_prec + full_recall+0.000001)

        print(" frame acc: %.6f " %  frame_acc)
        print(" fe_prec: %.6f " % fe_prec)
        print(" fe_recall: %.6f " % fe_recall)
        print(" fe_f1: %.6f " % fe_f1)
        print('================full struction=============')
        print(" full_prec: %.6f " % full_prec)
        print(" full_recall: %.6f " % full_recall)
        print(" full_f1: %.6f " % full_f1)
        print('================detail=============')
        print(" fe_TP: %.6f " % self.fe_TP)
        print(" fe_TP_FN: %.6f " % self.fe_TP_FN)
        print(" fe_TP_FP: %.6f " % self.fe_TP_FP)
        print(" core_cnt: %.6f " % self.core_cnt)
        print(" nocore_cnt: %.6f " % self.nocore_cnt)

        return (frame_acc,fe_prec,fe_recall,fe_f1,full_prec,full_recall,full_f1)