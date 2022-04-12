import xml.etree.cElementTree as ET
import re
import os
import numpy as np

from config import get_opt
from data_preprocess import load_data_pd, get_frame_table, get_fe_table

# path = '../data/fndata-1.5/frame/'

def parse_fe_xml(xml_dir='../data/fndata-1.5/frame/', filename='Absorb_heat.xml', tag_prefix = '{http://framenet.icsi.berkeley.edu}'):
    tree = ET.ElementTree(file=xml_dir + filename)
    root = tree.getroot()
    frame_name = root.attrib.get('name')

    frame_id = int(root.attrib.get('ID'))

    fe_set = set()
    fe_adj = {}
    # <FE> </FE>
    for fe in root:
        # child.tag/text
        if fe.tag == tag_prefix + 'FE':
            fe_name = fe.attrib.get('name')
            fe_set.add(fe_name)
        
    # print(fe_set)
    for fe in root:
        if fe.tag == tag_prefix + 'FE':
            fe_name = fe.attrib.get('name')
            fe_id = int(fe.attrib.get('ID'))
            fe_adj[(fe_name, fe_id)] = set()
            for defn in fe:
                if defn.tag == tag_prefix + 'definition':
                    word_list = re.sub("<.*?>", " ", defn.text).strip().split(' ')
                    for word in word_list:
                        if word != fe_name and word in fe_set:
                            fe_adj[(fe_name,  fe_id)].add(word)
                    # print(fe_name, re.sub("<.*?>", " ", defn.text).strip().split(' '))
    # print(fe_adj)
    return frame_name, frame_id, fe_adj

def build_fe_adj(data_path, xml_dir):
    all_fe_adj = {} # fe label -> list[fe label]
    frame_id_to_label, frame_name_to_label, \
    frame_name_to_id = get_frame_table(data_path, 'frame.csv')

    fe_id_to_label, fe_name_to_label, fe_name_to_id, \
    fe_id_to_type = get_fe_table(data_path, 'FE.csv')
    cnt = 0
    for file_name in os.listdir(xml_dir):
        if file_name == 'frame.xsl':
            continue
        frame_name, frame_id, fe_adj = parse_fe_xml(xml_dir, file_name)
        if frame_id != frame_name_to_id[frame_name]:
            print(f"error! {frame_id} {frame_name}")
        for k, v in fe_adj.items():
            fe_name, fe_id = k
            if fe_name_to_id[(fe_name, frame_id)] != fe_id:
                print(f"error! {frame_id} {fe_name} {fe_id} {fe_name_to_id[(fe_name, frame_id)]}")
            fe_label = fe_name_to_label[(fe_name, frame_id)]
            if len(v) == 0:
                continue
            all_fe_adj[fe_label] = []
            for to_fe_name in v:
                to_fe_label = fe_name_to_label[(to_fe_name, frame_id)]
                all_fe_adj[fe_label].append(to_fe_label)
            
        cnt += 1
        if cnt % 100 == 0:
            print(cnt)
    # print(all_fe_adj)
    np.save('../data/intra_frame_fe_relations.npy', all_fe_adj, allow_pickle=True)
    return

if __name__ == '__main__':
    build_fe_adj('../data/parsed-v1.5/', '../data/fndata-1.5/frame/')
    