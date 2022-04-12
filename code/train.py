import torch
import torch.nn as nn
import os
import multiprocessing as mp

from dataset import FrameNetDataset, FrameNetDataLoader
from utils import seed_everything
from evaluate import Eval
from config import get_opt
from data_preprocess import DataConfig
from model import Model


def evaluate(opt, model, dataset, best_metrics=None, show_case=False):
    model.eval()
    print('begin eval')
    evaler = Eval(opt, dataset)
    with torch.no_grad():
        test_dl = FrameNetDataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=True
        )

        for b in test_dl:
            word_ids = b['word']
            lemma_ids = b['lemma']
            pos_ids = b['pos']
            lengths = b['length']
            mask = b['mask']
            target_head = b['target_head']
            target_tail = b['target_tail']
            target_type = b['target_type']
            fe_head = b['fe_head']
            fe_tail = b['fe_tail']
            fe_type = b['fe_type']
            fe_cnt = b['fe_cnt']
            lu_mask = b['lu_mask']
            gold_frame_id = b['gold_frame_id']
            lu_frame_num = b['lu_frame_num']
            frame_list = b['frame_list']
            frame_fe_num = b['frame_fe_num']
            fe_list = b['fe_list']
            token_type_ids = b['token_type']
            sent_length = b['sent_length']
            target_mask_ids = b['target_mask']
            adj_list = b['adj_list']
            dep_A = b['dep']
            if opt.eval == 'full':
                return_dic = model(word_ids, lemma_ids, pos_ids, lengths, (target_head, target_tail), token_type_ids, mask, gold_frame_id, frame_list, fe_list, adj_list, dep_A, lu_mask )
            else:
                return_dic = model(word_ids, lemma_ids, pos_ids, lengths, (target_head, target_tail), token_type_ids, mask, gold_frame_id, frame_list, fe_list, adj_list, dep_A)
            # print(return_dic)
            evaler.metrics(batch_size=opt.batch_size, fe_cnt=fe_cnt, gold_fe_type=fe_type, gold_fe_head=fe_head, \
                           gold_fe_tail=fe_tail, gold_frame_type=target_type,
                           pred_fe_type=return_dic['pred_role_action'],
                           pred_fe_head=return_dic['pred_head_action'],
                           pred_fe_tail=return_dic['pred_tail_action'],
                           pred_frame_type=return_dic['pred_frame_action'],
                           fe_coretype=dataset.fe_coretype_table,sent_length=sent_length)
            # break

            if show_case:
                print('target head = ', target_head)
                print('target tail = ', target_tail)
                print('gold_fe_label = ', fe_type)
                print('pred_fe_label = ', return_dic['pred_role_action'])
                print('gold_head_label = ', fe_head)
                print('pred_head_label = ', return_dic['pred_head_action'])
                print('gold_tail_label = ', fe_tail)
                print('pred_tail_label = ', return_dic['pred_tail_action'])
            # break
        metrics = evaler.calculate()
            

        if best_metrics:

            if metrics[-1] > best_metrics:
                best_metrics = metrics[-1]

                torch.save(model.state_dict(), opt.save_model_path)

            return best_metrics

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    mp.set_start_method('spawn')
    torch.autograd.set_detect_anomaly(True)

    opt = get_opt()
    config = DataConfig(opt)

    if torch.cuda.is_available():
        device = torch.device(opt.cuda)
    else:
        device = torch.device('cpu')

    seed_everything(opt.seed)
    epochs = opt.epochs
    model = Model(opt, config)
    model.to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.6)

    frame_criterion =nn.CrossEntropyLoss()
    head_criterion = nn.CrossEntropyLoss()
    tail_criterion = nn.CrossEntropyLoss()
    fe_type_criterion = nn.CrossEntropyLoss()

    best_metric = -1

    if opt.mode == 'train':
        if os.path.exists(opt.pretrain_model_path) is True:
            print('load pretrain model...')
            model.load_state_dict(torch.load(opt.pretrain_model_path))
        train_dataset = FrameNetDataset(opt, config, config.train_instance_dic, device)
        dev_dataset = FrameNetDataset(opt, config, config.dev_instance_dic, device)
        # test_dataset = FrameNetDataset(opt, config, config.test_instance_dic, device)
        train_dl = FrameNetDataLoader(
                    train_dataset,
                    batch_size=opt.batch_size,
                    shuffle=True
                )
        for epoch in range(epochs):
            scheduler.step()
            model.train()
            step = 0
            sum_loss = 0
            cnt = 0
            print('==========epochs= ' + str(epoch))
            for b in train_dl:
                optimizer.zero_grad()
                word_ids = b['word']
                lemma_ids = b['lemma']
                pos_ids = b['pos']
                lengths = b['length']
                mask = b['mask']
                target_head = b['target_head']
                target_tail = b['target_tail']
                target_type = b['target_type']
                fe_head = b['fe_head']
                fe_tail = b['fe_tail']
                fe_type = b['fe_type']
                fe_cnt = b['fe_cnt']
                lu_mask = b['lu_mask']
                gold_frame_id = b['gold_frame_id']
                lu_frame_num = b['lu_frame_num']
                frame_list = b['frame_list']
                frame_fe_num = b['frame_fe_num']
                fe_list = b['fe_list']
                token_type_ids = b['token_type']
                sent_length = b['sent_length']
                target_mask_ids = b['target_mask']
                adj_list = b['adj_list']
                dep_A = b['dep']
                # dep_list = b['dep']
                # print(dep_list)
                return_dic = model(word_ids, lemma_ids, pos_ids, lengths, (target_head, target_tail), token_type_ids, mask, gold_frame_id, frame_list, fe_list, adj_list, dep_A)
                frame_loss = 0
                head_loss = 0
                tail_loss = 0
                type_loss = 0

                for batch_index in range(opt.batch_size):
                    pred_frame_label = return_dic['pred_frame_list'][0][batch_index].unsqueeze(0)

                    gold_frame_label = target_type[batch_index]

                    frame_loss += frame_criterion(pred_frame_label, gold_frame_label)
                    for fe_index in range(opt.fe_padding_num):
                        pred_type_label = return_dic['pred_role_list'][fe_index].squeeze()
                        pred_type_label = pred_type_label[batch_index].unsqueeze(0)

                        gold_type_label = fe_type[batch_index][fe_index].unsqueeze(0)
                        type_loss += fe_type_criterion(pred_type_label, gold_type_label)


                        if fe_index >= fe_cnt[batch_index]:
                            break

                        pred_head_label = return_dic['pred_head_list'][fe_index].squeeze()
                        pred_head_label = pred_head_label[batch_index].unsqueeze(0)

                        gold_head_label = fe_head[batch_index][fe_index].unsqueeze(0)
                        #    print(gold_head_label.size())
                        #    print(pred_head_label.size())
                        head_loss += head_criterion(pred_head_label, gold_head_label)

                        pred_tail_label = return_dic['pred_tail_list'][fe_index].squeeze()
                        pred_tail_label = pred_tail_label[batch_index].unsqueeze(0)

                        gold_tail_label = fe_tail[batch_index][fe_index].unsqueeze(0)
                        tail_loss += tail_criterion(pred_tail_label, gold_tail_label)

                loss = (0.1 * frame_loss + 0.3 * type_loss + 0.3 * head_loss + 0.3 * tail_loss) / (opt.batch_size)
                loss.backward()
                optimizer.step()
                step += 1
                if step % 20 == 0:
                    print(" | batch loss: %.6f step = %d" % (loss.item(), step))
                sum_loss += loss.item()
            print('| epoch %d  avg loss = %.6f' % (epoch, sum_loss / step))
            # eval
            best_metric = evaluate(opt, model, dev_dataset, best_metric)

    # test
    else:
        model.load_state_dict(torch.load(opt.save_model_path))
        test_dataset = FrameNetDataset(opt, config, config.test_instance_dic, device)
        evaluate(opt, model, test_dataset)

