from featurizer.database import Database
from featurizer.acollector import AttributesValCollector, total_node
import os
import numpy as np
import sys
sys.path.append("/luh/snuba_experiment/reef/program_synthesis")
sys.path.append("/luh/snuba_experiment/reef")
from heuristic_generator import HeuristicGenerator
import pickle

import argparse
from random import sample
import json
from collections import defaultdict
from document import Document

parser = argparse.ArgumentParser()
parser.add_argument("--dataset")
parser.add_argument("--gt")
parser.add_argument("--example", type=int)
args = parser.parse_args()

# print(args.dataset)

op_folder = "/luh/synthesis_data/%s/example/total" % args.dataset
attributes = ['COL_END', 'COL_START', 'NER', 'POS', 'POSITION', 'ROW_END', 'ROW_START', 'KEYWORD']

custume = [
    ('in_row_pob', "select ir.source from IN_ROW ir, TEXT t where ir.target = t.id AND t.value = 'Place of Birth' "),
    ('in_row_bp', "select ir.source from IN_ROW ir, TEXT t where ir.target = t.id AND t.value = 'Birth place' "),
    ('in_row_ic', "select ir.source from IN_ROW ir, TEXT t where ir.target = t.id AND t.value LIKE '%CURRENT%' "),
    ('in_row_ic', "select ir.source from IN_ROW ir, TEXT t where ir.target = t.id AND t.value LIKE '%Current%' "),    
    ('in_row_c', "select ir.source from IN_ROW ir, TEXT t where ir.target = t.id AND t.value LIKE '%Continuous%' ")
]

acollector = AttributesValCollector(op_folder, attributes, custume, args.dataset)
attributes_dict = acollector.get_attributes()


train_folder = "/luh/synthesis_data_snuba/%s/example/total/train" % args.dataset


with open("/luh/synthesis_data_snuba/ground_truth/cell/%s.json" % args.dataset, 'r') as f:
    gt_id_cell = set()
    for i in json.load(f):
        for j in i['example']:
            gt_id_cell.add(j)



with open("/luh/synthesis_data_snuba/ground_truth/span/%s.json" % args.dataset, 'r') as f:
    gt_id_span = set()
    for i in json.load(f):
        for j in i['example']:
            gt_id_span.add(j)


gt_id_val = gt_id_cell+gt_id_span




def get_train_primitives_ground_truth_without_samplimg():

    try:
        with open("train_primitive_%s.pkl" % args.dataset) as f:
            primitives = pickle.load(f)
        with open("train_primitive_%s_node.pkl" % args.dataset) as f:
            node_id = pickle.load(f)
        with open("train_docs_%s.pkl" % args.dataset) as f:
            docs = pickle.load(f)

    except IOError:
        start = 0
        end = 0
        number_nodes = total_node(train_folder)
        primitives = None
        node_id = np.zeros((number_nodes, ))

 
        docs = []

        for idx, f in enumerate(os.listdir(train_folder)[:3]):

            path = os.path.join(train_folder, f)
            
            d = Database(path, attributes, custume)
            prim, i = d.generate_primitive_matrix(attributes_dict)
            
            if primitives is None:
                primitives = np.zeros((number_nodes, prim.shape[1]))
                print(primitives.shape)
            
            end += prim.shape[0]
            
            primitives[start:end, :] = prim
            node_id[start:end] = i

            start = end

            doc = Document(f)
            
            for node_id_single in i:
                if node_id_single in gt_id_val:
                    doc.add_to_pos(node_id, gt_id_cell, gt_id_span)
                else:
                    doc.add_to_neg(node_id_single)
            
            docs.append(doc)

        
        with open("train_primitive_%s.pkl" % args.dataset, 'wb') as f:
            pickle.dump(primitives, f)
        with open("train_primitive_%s_node.pkl" % args.dataset, 'wb') as f:
            pickle.dump(node_id, f)
        with open("train_docs_%s.pkl" % args.dataset, 'wb') as f:
            pickle.dump(docs, f)

    ground = []
    for i in node_id:
        if i in gt_id_val:
            ground.append(1)
        else:
            ground.append(-1)
    ground = np.array(ground)

    return primitives, ground, docs, node_id


if args.example == -1:
    primitives, ground _, _ = get_train_primitives_ground_truth_without_samplimg()
    try:
        with open("train_feature_%s.pkl" % args.dataset, "rb") as f:
            pass
    except IOError:
        with open("train_feature_%s.pkl" % args.dataset, "wb") as f:
            print("size of train:", primitives.shape)
            pickle.dump(primitives, f)    

    with open("train_label_%s.pkl" % args.dataset, "wb") as f:
        pickle.dump(ground, f)

else:
    primitives, ground, docs, node_id = get_train_primitives_ground_truth_without_samplimg()

    selected_id = set()
    for d_i in sample(docs, args.example):
        pos, neg = d_i.get_sample()
        selected_id |= pos
        selected_id |= neg
    
    cropped_primitives = np.zeros((len(selected_id), primitives.shape[1]))
    cropped_ground = np.zeros((len(selected_id,)))

    for i, v in enumerate(selected_id):
        idx, = np.where(node_id == i)
        cropped_primitives[i:] = primitives[idx:]
        cropped_ground[i] = ground[idx]
    
    with open("train_feature_%s.pkl" % args.dataset, "wb") as f:
        print("size of train:", primitives.shape)
        pickle.dump(primitives, f)

    with open("train_label_%s.pkl" % args.dataset, "wb") as f:
        pickle.dump(ground, f)

# assert False

test_folder = "/luh/synthesis_data_snuba/%s/example/total/test" % args.dataset
start = 0
end = 0
number_nodes = total_node(test_folder)
primitives = None
node_id = np.zeros((number_nodes, ))

try:
    with open("test_primitive_%s.pkl" % args.dataset) as f:
        primitives = pickle.load(f)
    with open("test_primitive_%s_node.pkl" % args.dataset) as f:
        node_id = pickle.load(f)

except IOError:
    for idx, f in enumerate(os.listdir(test_folder)[:3]):

        path = os.path.join(test_folder, f)
        
        d = Database(path, attributes, custume)
        prim, i = d.generate_primitive_matrix(attributes_dict)
        
        if primitives is None:
            primitives = np.zeros((number_nodes, prim.shape[1]))
            print(primitives.shape)
        
        end += prim.shape[0]
        
        primitives[start:end, :] = prim
        node_id[start:end] = i

        start = end
    
    with open("test_primitive_%s.pkl" % args.dataset, 'wb') as f:
        pickle.dump(primitives, f)
    with open("test_primitive_%s_node.pkl" % args.dataset, 'wb') as f:
        pickle.dump(node_id, f)


ground = []
for i in node_id:
    if i in gt_id_val:
        ground.append(1)
    else:
        ground.append(-1)
ground = np.array(ground)

try:
    with open("test_feature_%s.pkl" % args.dataset, "rb") as f:
        pass
except IOError:
    with open("test_feature_%s.pkl" % args.dataset, "wb") as f:
        print("size of test:", primitives.shape)
        pickle.dump(primitives, f)    

with open("test_label_%s.pkl" % args.dataset, "wb") as f:
    pickle.dump(ground, f)





# op_folder = "/luh/synthesis_data_snuba/%s/example/total/validation" % args.dataset
# start = 0
# end = 0
# number_nodes = total_node(op_folder)
# primitives = None
# node_id = np.zeros((number_nodes, ))

# try:
#     with open("val_primitive_%s.pkl" % args.dataset) as f:
#         primitives = pickle.load(f)
#     with open("val_primitive_%s_node.pkl" % args.dataset) as f:
#         node_id = pickle.load(f)

# except IOError:
#     for idx, f in enumerate(os.listdir(op_folder)):

#         path = os.path.join(op_folder, f)
        
#         d = Database(path, attributes, custume)
#         prim, i = d.generate_primitive_matrix(attributes_dict)
        
#         if primitives is None:
#             primitives = np.zeros((number_nodes, prim.shape[1]))
#             print(primitives.shape)
        
#         end += prim.shape[0]
        
#         primitives[start:end, :] = prim
#         node_id[start:end] = i

#         start = end
    
#     with open("val_primitive_%s.pkl" % args.dataset, 'wb') as f:
#         pickle.dump(primitives, f)
#     with open("val_primitive_%s_node.pkl" % args.dataset, 'wb') as f:
#         pickle.dump(node_id, f)

# gt_id_val = []

# for s in gt_id:
#     for i in s['example']:
#         gt_id_val.append(i)

# gt_id_val = set(gt_id_val)

# ground = []
# for i in node_id:
#     if i in gt_id_val:
#         ground.append(1)
#     else:
#         ground.append(-1)
# ground = np.array(ground)

# try:
#     with open("val_feature_%s.pkl" % args.dataset, "rb") as f:
#         pass
# except IOError:
#     with open("val_feature_%s.pkl" % args.dataset, "wb") as f:
#         pickle.dump(primitives, f)

# with open("val_label.pkl", "wb") as f:
#     pickle.dump(ground, f)