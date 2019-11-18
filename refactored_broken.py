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

parser = argparse.ArgumentParser()
parser.add_argument("--example", type=int)
parser.add_argument("--dataset")
parser.add_argument("--gt")
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


for group in ['train', 'test', 'val']:
    try:
        with open("%s_primitive_%s.pkl" % (group, args.dataset)) as f:
            primitives = pickle.load(f)
        with open("%s_primitive_%s_node.pkl" % (group, args.dataset)) as f:
            node_id = pickle.load(f)

    except IOError:
        db_folder = "/luh/synthesis_data_snuba/%s/example/total/%s" % (args.dataset, group)
        start = 0
        end = 0
        number_nodes = total_node(db_folder)
        primitives = None
        node_id = np.zeros((number_nodes, ))
        for idx, f in enumerate(os.listdir(db_folder)):

            path = os.path.join(db_folder, f)
            
            d = Database(path, attributes, custume)
            prim, i = d.generate_primitive_matrix(attributes_dict)
            
            if primitives is None:
                primitives = np.zeros((number_nodes, prim.shape[1]))
                # print(primitives.shape)
            
            end += prim.shape[0]
            
            primitives[start:end, :] = prim
            node_id[start:end] = i

            start = end
        
        with open("%s_primitive_%s.pkl" % (group, args.dataset), 'wb') as f:
            pickle.dump(primitives, f)
        with open("%s_primitive_%s_node.pkl" % (group, args.dataset), 'wb') as f:
            pickle.dump(node_id, f)

    with open(args.gt, 'r') as f:
        gt_id_raw = json.load(f)

    gt_id = []
    for s in gt_id_raw:
        for i in s['example']:
            gt_id.append(i)

    gt_id = set(gt_id)

    ground = []
    for i in node_id:
        if i in gt_id:
            ground.append(1)
        else:
            ground.append(-1)
    ground = np.array(ground)


    num_examples = args.example

    pos = sample(np.where(ground == 1)[0], num_examples)
    neg = sample(np.where(ground == -1)[0], num_examples)

    if group == 'train':
        cropped_primitives = np.zeros(( 2*num_examples , primitives.shape[1]))
        ground_truth = np.zeros((2*num_examples,))

        for idx, i in enumerate(pos[:num_examples]):
            cropped_primitives[idx, :] = primitives[i, :]
            ground_truth[idx] = ground[i]

        for idx, i in enumerate(neg[:num_examples]):
            cropped_primitives[idx+num_examples, :] = primitives[i, :]
            ground_truth[idx+num_examples] = ground[i]

        with open("train_feature.pkl", "wb") as f:
            # print("size of train:", cropped_primitives.shape)
            pickle.dump(cropped_primitives, f)
        with open("train_label.pkl", "wb") as f:
            pickle.dump(ground_truth, f)
    else:
        with open("%s_feature.pkl" % group, "wb") as f:
            # print("size of test:", primitives.shape)
            pickle.dump(primitives, f)
        with open("%s_label.pkl" % group, "wb") as f:
            pickle.dump(ground, f)


# assert False

# test_folder = "/luh/synthesis_data_snuba/%s/example/total/test" % args.dataset
# start = 0
# end = 0
# number_nodes = total_node(test_folder)
# primitives = None
# node_id = np.zeros((number_nodes, ))

# try:
#     with open("test_primitive_%s.pkl" % args.dataset) as f:
#         primitives = pickle.load(f)
#     with open("test_primitive_%s_node.pkl" % args.dataset) as f:
#         node_id = pickle.load(f)

# except IOError:
#     for idx, f in enumerate(os.listdir(test_folder)):

#         path = os.path.join(test_folder, f)
        
#         d = Database(path, attributes, custume)
#         prim, i = d.generate_primitive_matrix(attributes_dict)
        
#         if primitives is None:
#             primitives = np.zeros((number_nodes, prim.shape[1]))
#             print(primitives.shape)
        
#         end += prim.shape[0]
        
#         primitives[start:end, :] = prim
#         node_id[start:end] = i

#         start = end
    
#     with open("test_primitive_%s.pkl" % args.dataset, 'wb') as f:
#         pickle.dump(primitives, f)
#     with open("test_primitive_%s_node.pkl" % args.dataset, 'wb') as f:
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


# with open("test_feature.pkl", "wb") as f:
#     print("size of test:", primitives.shape)
#     pickle.dump(primitives, f)
# with open("test_label.pkl", "wb") as f:
#     pickle.dump(ground, f)




# op_folder = "/luh/synthesis_data_snuba/%s/example/validation" % args.dataset
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

#     gt_id_val = []

#     for s in gt_id:
#         for i in s['example']:
#             gt_id_val.append(i)

#     gt_id_val = set(gt_id_val)

#     ground = []
#     for i in node_id:
#         if i in gt_id_val:
#             ground.append(1)
#         else:
#             ground.append(-1)
#     ground = np.array(ground)


#     with open("val_feature.pkl", "wb") as f:
#         pickle.dump(primitives, f)
#     with open("val_label.pkl", "wb") as f:
#         pickle.dump(ground, f)