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
start = 0
end = 0
number_nodes = total_node(train_folder)
primitives = None
node_id = np.zeros((number_nodes, ))


try:
    with open("train_primitive_%s.pkl" % args.dataset) as f:
        primitives = pickle.load(f)
    with open("train_primitive_%s_node.pkl" % args.dataset) as f:
        node_id = pickle.load(f)

except IOError:
    for idx, f in enumerate(os.listdir(train_folder)):

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
    
    with open("train_primitive_%s.pkl" % args.dataset, 'wb') as f:
        pickle.dump(primitives, f)
    with open("train_primitive_%s_node.pkl" % args.dataset, 'wb') as f:
        pickle.dump(node_id, f)

with open(args.gt, 'r') as f:
    gt_obj = json.load(f)

gt_id_val = []

for s in gt_obj:
    for i in s['example']:
        gt_id_val.append(i)

gt_id_val = set(gt_id_val)

ground = []
for i in node_id:
    if i in gt_id_val:
        ground.append(1)
    else:
        ground.append(-1)
ground = np.array(ground)


num_examples = args.example

list_of_pos_examples = []

for n in gt_obj:
	if n['file'].split('/')[-1] in set(os.listdir(train_folder)):
		list_of_pos_examples.append(n['example'])


if (num_examples > len(list_of_pos_examples)):
    num_examples = len(list_of_pos_examples)

    with open("results_log.csv", "a") as f:
        f.write("\nExceed positive examples in train set. Truncate to %d" % num_examples)

label_count = dict(zip(*np.unique(ground, return_counts=True)))

# print(np.where(ground == 1)[0])
# print(num_examples)
# assert False
if args.example == -1:
    with open("train_feature_%s.pkl" % args.dataset, "wb") as f:
        print("size of train:", primitives.shape)
        pickle.dump(primitives, f)
    with open("train_label_%s.pkl" % args.dataset, "wb") as f:
        pickle.dump(ground, f)
    with open("num_pos_example_%s.pkl" % args.dataset, "wb") as f:
        pickle.dump(len(list_of_pos_examples), f)
else:
    pos = []
    for i in sample(list_of_pos_examples, num_examples):
        for i_j in i:
            idx, = np.where(node_id == i_j)[0]
            pos.append(idx)

    neg = sample(np.where(ground == -1)[0], int(num_examples * float(label_count[-1]/len(list_of_pos_examples))))

    data_points_num = len(pos)+len(neg)

    cropped_primitives = np.zeros((data_points_num, primitives.shape[1]))
    ground_truth = np.zeros((data_points_num))

    # print(pos)
    for idx, i in enumerate(pos):
        cropped_primitives[idx, :] = primitives[i, :]
        ground_truth[idx] = ground[i]

    for idx, i in enumerate(neg):
        cropped_primitives[idx+num_examples, :] = primitives[i, :]
        ground_truth[idx+num_examples] = ground[i]

    with open("train_feature_%s.pkl" % args.dataset, "wb") as f:
        print("size of train:", cropped_primitives.shape)
        pickle.dump(cropped_primitives, f)
    with open("train_label_%s.pkl" % args.dataset, "wb") as f:
        pickle.dump(ground_truth, f)

# print(dict(zip(*np.unique(ground_truth, return_counts=True))))
# assert False

# try:
#     with open("train_feature_%s.pkl" % args.dataset, "rb") as f:
#         pass
# except IOError:
#     with open("train_feature_%s.pkl" % args.dataset, "wb") as f:
#         print("size of train:", primitives.shape)
#         pickle.dump(primitives, f)    


# with open("train_label_%s.pkl" % args.dataset, "wb") as f:
#     pickle.dump(ground, f) 


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
    for idx, f in enumerate(os.listdir(test_folder)):

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


gt_id_val = []

for s in gt_obj:
    for i in s['example']:
        gt_id_val.append(i)

gt_id_val = set(gt_id_val)

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