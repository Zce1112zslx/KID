import numpy as np
import stanza


from config import get_opt
# from data_preprocess import DataConfig


def padding_sentence(sentence: list,maxlen: int):
    while len(sentence) < maxlen:
        sentence.append('<pad>')

    return sentence

def dep_parsing(word_list: list, max_len: int, parser):
    sent_raw = ' '.join(word_list)
    sents = parser(sent_raw).sentences
    token_list = []
    head_list = []
    rel_list = []
    for sent in sents:
    # sent = parser(sent_raw).sentences[0]

        token_list += [word.text for word in sent.words]
        head_list += [word.head - 1 if word.head != 0 else i for i, word in enumerate(sent.words)]
        rel_list += [word.deprel for word in sent.words]
    if (len(head_list) != len(word_list)):
        print(sent_raw)
        word_set = set(word_list)
        token_set = set(token_list)
        same_set = word_set & token_set
        print(word_set - same_set, token_set - same_set)
    length = len(word_list)
    
    while length < max_len:
        head_list.append(length)
        rel_list.append('<pad>')
        length += 1
    return head_list, rel_list

def process_instance(instance, maxlen, parser, save_path):
    cnt = 0
    for k, line in instance.items():
        length = line['length']
        instance[k]['dep_list'] = dep_parsing(line['word_list'][:length], maxlen, parser)
        cnt += 1
        if cnt % 200 == 0:
            print(cnt)
    np.save(save_path, instance)
def process(opt):
    config = {
        'lang': 'en', 
        'tokenize_pretokenized':True,
    }
    parser = stanza.Pipeline(**config)
    maxlen = opt.maxlen
    exemplar_instance_dic = np.load(opt.exemplar_instance_path, allow_pickle=True).item()
    train_instance_dic = np.load(opt.train_instance_path, allow_pickle=True).item()
    dev_instance_dic = np.load(opt.dev_instance_path, allow_pickle=True).item()
    test_instance_dic = np.load(opt.test_instance_path, allow_pickle=True).item()
    print("begin train")
    process_instance(train_instance_dic, maxlen, parser, opt.train_instance_path)
    print("begin dev")
    process_instance(dev_instance_dic, maxlen, parser, opt.dev_instance_path)
    print("begin test")
    process_instance(test_instance_dic, maxlen, parser, opt.test_instance_path)
    print("begin exemplar")
    process_instance(exemplar_instance_dic, maxlen, parser, opt.exemplar_instance_path)


        


if __name__ == '__main__':
    opt = get_opt()
    process(opt)
    # opt.load_instance_dic = False
    # config = DataConfig(opt)
    # print(config.word_vectors)
    # print(config.lemma_number)
    # print(config.word_number)
    # print(config.pos_number)