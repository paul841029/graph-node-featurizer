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

# parser = argparse.ArgumentParser()
# parser.add_argument("--test_rate", type=float)
# args = parser.parse_args()

op_folder = "/luh/synthesis_data_snuba/political_table/example/total"
attributes = ['COL_END', 'COL_START', 'NER', 'POS', 'POSITION', 'ROW_END', 'ROW_START', 'KEYWORD']

custume = [
    ('in_row_pob', "select ir.source from IN_ROW ir, TEXT t where ir.target = t.id AND t.value = 'Place of Birth' "),
    ('in_row_bp', "select ir.source from IN_ROW ir, TEXT t where ir.target = t.id AND t.value = 'Birth place' "),
    ('in_row_ic', "select ir.source from IN_ROW ir, TEXT t where ir.target = t.id AND t.value = 'Ic' ")
]

acollector = AttributesValCollector(op_folder, attributes, custume, 'political')
attributes_dict = acollector.get_attributes()



train_folder = "/luh/synthesis_data_snuba/political_table/example/total/train"
start = 0
end = 0
number_nodes = total_node(train_folder)
primitives = None
node_id = np.zeros((number_nodes, ))

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

gt_id =\
    [
    {
        "file": "/data/39.db", 
        "example": [
            2788
        ]
    }, 
    {
        "file": "/data/53.db", 
        "example": [
            2602
        ]
    }, 
    {
        "file": "/data/75.db", 
        "example": [
            3242
        ]
    }, 
    {
        "file": "/data/73.db", 
        "example": [
            4595
        ]
    }, 
    {
        "file": "/data/1.db", 
        "example": [
            1487
        ]
    }, 
    {
        "file": "/data/50.db", 
        "example": [
            3767
        ]
    }, 
    {
        "file": "/data/81.db", 
        "example": [
            3899
        ]
    }, 
    {
        "file": "/data/40.db", 
        "example": [
            1925
        ]
    }, 
    {
        "file": "/data/91.db", 
        "example": [
            4072
        ]
    }, 
    {
        "file": "/data/21.db", 
        "example": [
            1257
        ]
    }, 
    {
        "file": "/data/11.db", 
        "example": [
            952
        ]
    }, 
    {
        "file": "/data/103.db", 
        "example": [
            1011
        ]
    }, 
    {
        "file": "/data/71.db", 
        "example": [
            3412
        ]
    }, 
    {
        "file": "/data/7.db", 
        "example": [
            3076
        ]
    }, 
    {
        "file": "/data/90.db", 
        "example": [
            4928
        ]
    }, 
    {
        "file": "/data/28.db", 
        "example": [
            2628
        ]
    }, 
    {
        "file": "/data/47.db", 
        "example": [
            2256
        ]
    }, 
    {
        "file": "/data/38.db", 
        "example": [
            3112
        ]
    }, 
    {
        "file": "/data/54.db", 
        "example": [
            3934
        ]
    }, 
    {
        "file": "/data/77.db", 
        "example": [
            3748
        ]
    }, 
    {
        "file": "/data/18.db", 
        "example": [
            1175
        ]
    }, 
    {
        "file": "/data/92.db", 
        "example": [
            4224
        ]
    }, 
    {
        "file": "/data/17.db", 
        "example": [
            1052
        ]
    }, 
    {
        "file": "/data/25.db", 
        "example": [
            1299
        ]
    }, 
    {
        "file": "/data/74.db", 
        "example": [
            3579
        ]
    }, 
    {
        "file": "/data/9.db", 
        "example": [
            4925
        ]
    }, 
    {
        "file": "/data/82.db", 
        "example": [
            4764
        ]
    }, 
    {
        "file": "/data/0.db", 
        "example": [
            891
        ]
    }, 
    {
        "file": "/data/69.db", 
        "example": [
            4269
        ]
    }, 
    {
        "file": "/data/64.db", 
        "example": [
            3085
        ]
    }, 
    {
        "file": "/data/22.db", 
        "example": [
            1217
        ]
    }, 
    {
        "file": "/data/61.db", 
        "example": [
            2924
        ]
    }, 
    {
        "file": "/data/35.db", 
        "example": [
            2956
        ]
    }, 
    {
        "file": "/data/24.db", 
        "example": [
            2472
        ]
    }, 
    {
        "file": "/data/100.db", 
        "example": [
            740
        ]
    }, 
    {
        "file": "/data/96.db", 
        "example": [
            4382
        ]
    }, 
    {
        "file": "/data/94.db", 
        "example": [
            5010
        ]
    }, 
    {
        "file": "/data/95.db", 
        "example": [
            5088
        ]
    }, 
    {
        "file": "/data/65.db", 
        "example": [
            4109
        ]
    }, 
    {
        "file": "/data/3.db", 
        "example": [
            1456
        ]
    }, 
    {
        "file": "/data/4.db", 
        "example": [
            1944
        ]
    }, 
    {
        "file": "/data/62.db", 
        "example": [
            3950
        ]
    }, 
    {
        "file": "/data/70.db", 
        "example": [
            4514
        ]
    }, 
    {
        "file": "/data/86.db", 
        "example": [
            4845
        ]
    }, 
    {
        "file": "/data/68.db", 
        "example": [
            4409
        ]
    }, 
    {
        "file": "/data/51.db", 
        "example": [
            2415
        ]
    }, 
    {
        "file": "/data/101.db", 
        "example": [
            838
        ]
    }, 
    {
        "file": "/data/30.db", 
        "example": [
            2796
        ]
    }, 
    {
        "file": "/data/37.db", 
        "example": [
            1758
        ]
    }, 
    {
        "file": "/data/8.db", 
        "example": [
            3404
        ]
    }, 
    {
        "file": "/data/63.db", 
        "example": [
            4259
        ]
    }, 
    {
        "file": "/data/58.db", 
        "example": [
            3778
        ]
    }, 
    {
        "file": "/data/43.db", 
        "example": [
            3446
        ]
    }, 
    {
        "file": "/data/5.db", 
        "example": [
            3297
        ]
    }, 
    {
        "file": "/data/98.db", 
        "example": [
            5163
        ]
    }, 
    {
        "file": "/data/44.db", 
        "example": [
            2116
        ]
    }, 
    {
        "file": "/data/26.db", 
        "example": [
            1346
        ]
    }, 
    {
        "file": "/data/52.db", 
        "example": [
            2446
        ]
    }, 
    {
        "file": "/data/66.db", 
        "example": [
            3244
        ]
    }, 
    {
        "file": "/data/10.db", 
        "example": [
            1352
        ]
    }, 
    {
        "file": "/data/16.db", 
        "example": [
            2135
        ]
    }, 
    {
        "file": "/data/60.db", 
        "example": [
            4097
        ]
    }, 
    {
        "file": "/data/89.db", 
        "example": [
            3898
        ]
    }, 
    {
        "file": "/data/13.db", 
        "example": [
            1799
        ]
    }, 
    {
        "file": "/data/48.db", 
        "example": [
            3606
        ]
    }, 
    {
        "file": "/data/6.db", 
        "example": [
            2749
        ]
    }, 
    {
        "file": "/data/33.db", 
        "example": [
            1585
        ]
    }, 
    {
        "file": "/data/14.db", 
        "example": [
            1093
        ]
    }, 
    {
        "file": "/data/29.db", 
        "example": [
            1433
        ]
    }, 
    {
        "file": "/data/23.db", 
        "example": [
            2126
        ]
    }, 
    {
        "file": "/data/56.db", 
        "example": [
            2585
        ]
    }, 
    {
        "file": "/data/84.db", 
        "example": [
            3572
        ]
    }, 
    {
        "file": "/data/55.db", 
        "example": [
            3455
        ]
    }, 
    {
        "file": "/data/67.db", 
        "example": [
            2913
        ]
    }, 
    {
        "file": "/data/87.db", 
        "example": [
            4062
        ]
    }, 
    {
        "file": "/data/105.db", 
        "example": [
            1637
        ]
    }, 
    {
        "file": "/data/27.db", 
        "example": [
            2293
        ]
    }, 
    {
        "file": "/data/79.db", 
        "example": [
            4680
        ]
    }, 
    {
        "file": "/data/31.db", 
        "example": [
            2449
        ]
    }, 
    {
        "file": "/data/49.db", 
        "example": [
            2277
        ]
    }, 
    {
        "file": "/data/102.db", 
        "example": [
            1647
        ]
    }, 
    {
        "file": "/data/2.db", 
        "example": [
            2299
        ]
    }, 
    {
        "file": "/data/99.db", 
        "example": [
            5093
        ]
    }, 
    {
        "file": "/data/42.db", 
        "example": [
            2955
        ]
    }, 
    {
        "file": "/data/34.db", 
        "example": [
            2618
        ]
    }, 
    {
        "file": "/data/72.db", 
        "example": [
            4417
        ]
    }, 
    {
        "file": "/data/78.db", 
        "example": [
            4602
        ]
    }, 
    {
        "file": "/data/97.db", 
        "example": [
            4235
        ]
    }, 
    {
        "file": "/data/36.db", 
        "example": [
            1787
        ]
    }, 
    {
        "file": "/data/83.db", 
        "example": [
            4761
        ]
    }, 
    {
        "file": "/data/76.db", 
        "example": [
            4519
        ]
    }, 
    {
        "file": "/data/46.db", 
        "example": [
            3132
        ]
    }, 
    {
        "file": "/data/85.db", 
        "example": [
            4844
        ]
    }, 
    {
        "file": "/data/19.db", 
        "example": [
            1967
        ]
    }, 
    {
        "file": "/data/32.db", 
        "example": [
            1625
        ]
    }
]

gt_id_val = []

for s in gt_id:
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


num_examples = 2

pos = sample(np.where(ground == 1)[0], num_examples)
neg = sample(np.where(ground == -1)[0], num_examples)


cropped_primitives = np.zeros(( 2*num_examples , primitives.shape[1]))
ground_truth = np.zeros((2*num_examples,))

for idx, i in enumerate(pos[:num_examples]):
    cropped_primitives[idx, :] = primitives[i, :]
    ground_truth[idx] = ground[i]

for idx, i in enumerate(neg[:num_examples]):
    cropped_primitives[idx+num_examples, :] = primitives[i, :]
    ground_truth[idx+num_examples] = ground[i]

with open("train_feature.pkl", "wb") as f:
    print("size of train:", cropped_primitives.shape)
    pickle.dump(cropped_primitives, f)
with open("train_label.pkl", "wb") as f:
    pickle.dump(ground_truth, f)

# assert False

test_folder = "/luh/synthesis_data_snuba/political_table/example/total/test"
start = 0
end = 0
number_nodes = total_node(test_folder)
primitives = None
node_id = np.zeros((number_nodes, ))

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

gt_id =\
    [
    {
        "file": "/data/39.db", 
        "example": [
            2788
        ]
    }, 
    {
        "file": "/data/53.db", 
        "example": [
            2602
        ]
    }, 
    {
        "file": "/data/75.db", 
        "example": [
            3242
        ]
    }, 
    {
        "file": "/data/73.db", 
        "example": [
            4595
        ]
    }, 
    {
        "file": "/data/1.db", 
        "example": [
            1487
        ]
    }, 
    {
        "file": "/data/50.db", 
        "example": [
            3767
        ]
    }, 
    {
        "file": "/data/81.db", 
        "example": [
            3899
        ]
    }, 
    {
        "file": "/data/40.db", 
        "example": [
            1925
        ]
    }, 
    {
        "file": "/data/91.db", 
        "example": [
            4072
        ]
    }, 
    {
        "file": "/data/21.db", 
        "example": [
            1257
        ]
    }, 
    {
        "file": "/data/11.db", 
        "example": [
            952
        ]
    }, 
    {
        "file": "/data/103.db", 
        "example": [
            1011
        ]
    }, 
    {
        "file": "/data/71.db", 
        "example": [
            3412
        ]
    }, 
    {
        "file": "/data/7.db", 
        "example": [
            3076
        ]
    }, 
    {
        "file": "/data/90.db", 
        "example": [
            4928
        ]
    }, 
    {
        "file": "/data/28.db", 
        "example": [
            2628
        ]
    }, 
    {
        "file": "/data/47.db", 
        "example": [
            2256
        ]
    }, 
    {
        "file": "/data/38.db", 
        "example": [
            3112
        ]
    }, 
    {
        "file": "/data/54.db", 
        "example": [
            3934
        ]
    }, 
    {
        "file": "/data/77.db", 
        "example": [
            3748
        ]
    }, 
    {
        "file": "/data/18.db", 
        "example": [
            1175
        ]
    }, 
    {
        "file": "/data/92.db", 
        "example": [
            4224
        ]
    }, 
    {
        "file": "/data/17.db", 
        "example": [
            1052
        ]
    }, 
    {
        "file": "/data/25.db", 
        "example": [
            1299
        ]
    }, 
    {
        "file": "/data/74.db", 
        "example": [
            3579
        ]
    }, 
    {
        "file": "/data/9.db", 
        "example": [
            4925
        ]
    }, 
    {
        "file": "/data/82.db", 
        "example": [
            4764
        ]
    }, 
    {
        "file": "/data/0.db", 
        "example": [
            891
        ]
    }, 
    {
        "file": "/data/69.db", 
        "example": [
            4269
        ]
    }, 
    {
        "file": "/data/64.db", 
        "example": [
            3085
        ]
    }, 
    {
        "file": "/data/22.db", 
        "example": [
            1217
        ]
    }, 
    {
        "file": "/data/61.db", 
        "example": [
            2924
        ]
    }, 
    {
        "file": "/data/35.db", 
        "example": [
            2956
        ]
    }, 
    {
        "file": "/data/24.db", 
        "example": [
            2472
        ]
    }, 
    {
        "file": "/data/100.db", 
        "example": [
            740
        ]
    }, 
    {
        "file": "/data/96.db", 
        "example": [
            4382
        ]
    }, 
    {
        "file": "/data/94.db", 
        "example": [
            5010
        ]
    }, 
    {
        "file": "/data/95.db", 
        "example": [
            5088
        ]
    }, 
    {
        "file": "/data/65.db", 
        "example": [
            4109
        ]
    }, 
    {
        "file": "/data/3.db", 
        "example": [
            1456
        ]
    }, 
    {
        "file": "/data/4.db", 
        "example": [
            1944
        ]
    }, 
    {
        "file": "/data/62.db", 
        "example": [
            3950
        ]
    }, 
    {
        "file": "/data/70.db", 
        "example": [
            4514
        ]
    }, 
    {
        "file": "/data/86.db", 
        "example": [
            4845
        ]
    }, 
    {
        "file": "/data/68.db", 
        "example": [
            4409
        ]
    }, 
    {
        "file": "/data/51.db", 
        "example": [
            2415
        ]
    }, 
    {
        "file": "/data/101.db", 
        "example": [
            838
        ]
    }, 
    {
        "file": "/data/30.db", 
        "example": [
            2796
        ]
    }, 
    {
        "file": "/data/37.db", 
        "example": [
            1758
        ]
    }, 
    {
        "file": "/data/8.db", 
        "example": [
            3404
        ]
    }, 
    {
        "file": "/data/63.db", 
        "example": [
            4259
        ]
    }, 
    {
        "file": "/data/58.db", 
        "example": [
            3778
        ]
    }, 
    {
        "file": "/data/43.db", 
        "example": [
            3446
        ]
    }, 
    {
        "file": "/data/5.db", 
        "example": [
            3297
        ]
    }, 
    {
        "file": "/data/98.db", 
        "example": [
            5163
        ]
    }, 
    {
        "file": "/data/44.db", 
        "example": [
            2116
        ]
    }, 
    {
        "file": "/data/26.db", 
        "example": [
            1346
        ]
    }, 
    {
        "file": "/data/52.db", 
        "example": [
            2446
        ]
    }, 
    {
        "file": "/data/66.db", 
        "example": [
            3244
        ]
    }, 
    {
        "file": "/data/10.db", 
        "example": [
            1352
        ]
    }, 
    {
        "file": "/data/16.db", 
        "example": [
            2135
        ]
    }, 
    {
        "file": "/data/60.db", 
        "example": [
            4097
        ]
    }, 
    {
        "file": "/data/89.db", 
        "example": [
            3898
        ]
    }, 
    {
        "file": "/data/13.db", 
        "example": [
            1799
        ]
    }, 
    {
        "file": "/data/48.db", 
        "example": [
            3606
        ]
    }, 
    {
        "file": "/data/6.db", 
        "example": [
            2749
        ]
    }, 
    {
        "file": "/data/33.db", 
        "example": [
            1585
        ]
    }, 
    {
        "file": "/data/14.db", 
        "example": [
            1093
        ]
    }, 
    {
        "file": "/data/29.db", 
        "example": [
            1433
        ]
    }, 
    {
        "file": "/data/23.db", 
        "example": [
            2126
        ]
    }, 
    {
        "file": "/data/56.db", 
        "example": [
            2585
        ]
    }, 
    {
        "file": "/data/84.db", 
        "example": [
            3572
        ]
    }, 
    {
        "file": "/data/55.db", 
        "example": [
            3455
        ]
    }, 
    {
        "file": "/data/67.db", 
        "example": [
            2913
        ]
    }, 
    {
        "file": "/data/87.db", 
        "example": [
            4062
        ]
    }, 
    {
        "file": "/data/105.db", 
        "example": [
            1637
        ]
    }, 
    {
        "file": "/data/27.db", 
        "example": [
            2293
        ]
    }, 
    {
        "file": "/data/79.db", 
        "example": [
            4680
        ]
    }, 
    {
        "file": "/data/31.db", 
        "example": [
            2449
        ]
    }, 
    {
        "file": "/data/49.db", 
        "example": [
            2277
        ]
    }, 
    {
        "file": "/data/102.db", 
        "example": [
            1647
        ]
    }, 
    {
        "file": "/data/2.db", 
        "example": [
            2299
        ]
    }, 
    {
        "file": "/data/99.db", 
        "example": [
            5093
        ]
    }, 
    {
        "file": "/data/42.db", 
        "example": [
            2955
        ]
    }, 
    {
        "file": "/data/34.db", 
        "example": [
            2618
        ]
    }, 
    {
        "file": "/data/72.db", 
        "example": [
            4417
        ]
    }, 
    {
        "file": "/data/78.db", 
        "example": [
            4602
        ]
    }, 
    {
        "file": "/data/97.db", 
        "example": [
            4235
        ]
    }, 
    {
        "file": "/data/36.db", 
        "example": [
            1787
        ]
    }, 
    {
        "file": "/data/83.db", 
        "example": [
            4761
        ]
    }, 
    {
        "file": "/data/76.db", 
        "example": [
            4519
        ]
    }, 
    {
        "file": "/data/46.db", 
        "example": [
            3132
        ]
    }, 
    {
        "file": "/data/85.db", 
        "example": [
            4844
        ]
    }, 
    {
        "file": "/data/19.db", 
        "example": [
            1967
        ]
    }, 
    {
        "file": "/data/32.db", 
        "example": [
            1625
        ]
    }
]

gt_id_val = []

for s in gt_id:
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




with open("test_feature.pkl", "wb") as f:
    print("size of test:", primitives.shape)
    pickle.dump(primitives, f)
with open("test_label.pkl", "wb") as f:
    pickle.dump(ground, f)




op_folder = "/luh/synthesis_data_snuba/political_table/example/validation"
start = 0
end = 0
number_nodes = total_node(op_folder)
primitives = None
node_id = np.zeros((number_nodes, ))

for idx, f in enumerate(os.listdir(op_folder)):

    path = os.path.join(op_folder, f)
    
    d = Database(path, attributes, custume)
    prim, i = d.generate_primitive_matrix(attributes_dict)
    
    if primitives is None:
        primitives = np.zeros((number_nodes, prim.shape[1]))
        print(primitives.shape)
    
    end += prim.shape[0]
    
    primitives[start:end, :] = prim
    node_id[start:end] = i

    start = end


gt_id =\
    [
    {
        "file": "/data/39.db", 
        "example": [
            2788
        ]
    }, 
    {
        "file": "/data/53.db", 
        "example": [
            2602
        ]
    }, 
    {
        "file": "/data/75.db", 
        "example": [
            3242
        ]
    }, 
    {
        "file": "/data/73.db", 
        "example": [
            4595
        ]
    }, 
    {
        "file": "/data/1.db", 
        "example": [
            1487
        ]
    }, 
    {
        "file": "/data/50.db", 
        "example": [
            3767
        ]
    }, 
    {
        "file": "/data/81.db", 
        "example": [
            3899
        ]
    }, 
    {
        "file": "/data/40.db", 
        "example": [
            1925
        ]
    }, 
    {
        "file": "/data/91.db", 
        "example": [
            4072
        ]
    }, 
    {
        "file": "/data/21.db", 
        "example": [
            1257
        ]
    }, 
    {
        "file": "/data/11.db", 
        "example": [
            952
        ]
    }, 
    {
        "file": "/data/103.db", 
        "example": [
            1011
        ]
    }, 
    {
        "file": "/data/71.db", 
        "example": [
            3412
        ]
    }, 
    {
        "file": "/data/7.db", 
        "example": [
            3076
        ]
    }, 
    {
        "file": "/data/90.db", 
        "example": [
            4928
        ]
    }, 
    {
        "file": "/data/28.db", 
        "example": [
            2628
        ]
    }, 
    {
        "file": "/data/47.db", 
        "example": [
            2256
        ]
    }, 
    {
        "file": "/data/38.db", 
        "example": [
            3112
        ]
    }, 
    {
        "file": "/data/54.db", 
        "example": [
            3934
        ]
    }, 
    {
        "file": "/data/77.db", 
        "example": [
            3748
        ]
    }, 
    {
        "file": "/data/18.db", 
        "example": [
            1175
        ]
    }, 
    {
        "file": "/data/92.db", 
        "example": [
            4224
        ]
    }, 
    {
        "file": "/data/17.db", 
        "example": [
            1052
        ]
    }, 
    {
        "file": "/data/25.db", 
        "example": [
            1299
        ]
    }, 
    {
        "file": "/data/74.db", 
        "example": [
            3579
        ]
    }, 
    {
        "file": "/data/9.db", 
        "example": [
            4925
        ]
    }, 
    {
        "file": "/data/82.db", 
        "example": [
            4764
        ]
    }, 
    {
        "file": "/data/0.db", 
        "example": [
            891
        ]
    }, 
    {
        "file": "/data/69.db", 
        "example": [
            4269
        ]
    }, 
    {
        "file": "/data/64.db", 
        "example": [
            3085
        ]
    }, 
    {
        "file": "/data/22.db", 
        "example": [
            1217
        ]
    }, 
    {
        "file": "/data/61.db", 
        "example": [
            2924
        ]
    }, 
    {
        "file": "/data/35.db", 
        "example": [
            2956
        ]
    }, 
    {
        "file": "/data/24.db", 
        "example": [
            2472
        ]
    }, 
    {
        "file": "/data/100.db", 
        "example": [
            740
        ]
    }, 
    {
        "file": "/data/96.db", 
        "example": [
            4382
        ]
    }, 
    {
        "file": "/data/94.db", 
        "example": [
            5010
        ]
    }, 
    {
        "file": "/data/95.db", 
        "example": [
            5088
        ]
    }, 
    {
        "file": "/data/65.db", 
        "example": [
            4109
        ]
    }, 
    {
        "file": "/data/3.db", 
        "example": [
            1456
        ]
    }, 
    {
        "file": "/data/4.db", 
        "example": [
            1944
        ]
    }, 
    {
        "file": "/data/62.db", 
        "example": [
            3950
        ]
    }, 
    {
        "file": "/data/70.db", 
        "example": [
            4514
        ]
    }, 
    {
        "file": "/data/86.db", 
        "example": [
            4845
        ]
    }, 
    {
        "file": "/data/68.db", 
        "example": [
            4409
        ]
    }, 
    {
        "file": "/data/51.db", 
        "example": [
            2415
        ]
    }, 
    {
        "file": "/data/101.db", 
        "example": [
            838
        ]
    }, 
    {
        "file": "/data/30.db", 
        "example": [
            2796
        ]
    }, 
    {
        "file": "/data/37.db", 
        "example": [
            1758
        ]
    }, 
    {
        "file": "/data/8.db", 
        "example": [
            3404
        ]
    }, 
    {
        "file": "/data/63.db", 
        "example": [
            4259
        ]
    }, 
    {
        "file": "/data/58.db", 
        "example": [
            3778
        ]
    }, 
    {
        "file": "/data/43.db", 
        "example": [
            3446
        ]
    }, 
    {
        "file": "/data/5.db", 
        "example": [
            3297
        ]
    }, 
    {
        "file": "/data/98.db", 
        "example": [
            5163
        ]
    }, 
    {
        "file": "/data/44.db", 
        "example": [
            2116
        ]
    }, 
    {
        "file": "/data/26.db", 
        "example": [
            1346
        ]
    }, 
    {
        "file": "/data/52.db", 
        "example": [
            2446
        ]
    }, 
    {
        "file": "/data/66.db", 
        "example": [
            3244
        ]
    }, 
    {
        "file": "/data/10.db", 
        "example": [
            1352
        ]
    }, 
    {
        "file": "/data/16.db", 
        "example": [
            2135
        ]
    }, 
    {
        "file": "/data/60.db", 
        "example": [
            4097
        ]
    }, 
    {
        "file": "/data/89.db", 
        "example": [
            3898
        ]
    }, 
    {
        "file": "/data/13.db", 
        "example": [
            1799
        ]
    }, 
    {
        "file": "/data/48.db", 
        "example": [
            3606
        ]
    }, 
    {
        "file": "/data/6.db", 
        "example": [
            2749
        ]
    }, 
    {
        "file": "/data/33.db", 
        "example": [
            1585
        ]
    }, 
    {
        "file": "/data/14.db", 
        "example": [
            1093
        ]
    }, 
    {
        "file": "/data/29.db", 
        "example": [
            1433
        ]
    }, 
    {
        "file": "/data/23.db", 
        "example": [
            2126
        ]
    }, 
    {
        "file": "/data/56.db", 
        "example": [
            2585
        ]
    }, 
    {
        "file": "/data/84.db", 
        "example": [
            3572
        ]
    }, 
    {
        "file": "/data/55.db", 
        "example": [
            3455
        ]
    }, 
    {
        "file": "/data/67.db", 
        "example": [
            2913
        ]
    }, 
    {
        "file": "/data/87.db", 
        "example": [
            4062
        ]
    }, 
    {
        "file": "/data/105.db", 
        "example": [
            1637
        ]
    }, 
    {
        "file": "/data/27.db", 
        "example": [
            2293
        ]
    }, 
    {
        "file": "/data/79.db", 
        "example": [
            4680
        ]
    }, 
    {
        "file": "/data/31.db", 
        "example": [
            2449
        ]
    }, 
    {
        "file": "/data/49.db", 
        "example": [
            2277
        ]
    }, 
    {
        "file": "/data/102.db", 
        "example": [
            1647
        ]
    }, 
    {
        "file": "/data/2.db", 
        "example": [
            2299
        ]
    }, 
    {
        "file": "/data/99.db", 
        "example": [
            5093
        ]
    }, 
    {
        "file": "/data/42.db", 
        "example": [
            2955
        ]
    }, 
    {
        "file": "/data/34.db", 
        "example": [
            2618
        ]
    }, 
    {
        "file": "/data/72.db", 
        "example": [
            4417
        ]
    }, 
    {
        "file": "/data/78.db", 
        "example": [
            4602
        ]
    }, 
    {
        "file": "/data/97.db", 
        "example": [
            4235
        ]
    }, 
    {
        "file": "/data/36.db", 
        "example": [
            1787
        ]
    }, 
    {
        "file": "/data/83.db", 
        "example": [
            4761
        ]
    }, 
    {
        "file": "/data/76.db", 
        "example": [
            4519
        ]
    }, 
    {
        "file": "/data/46.db", 
        "example": [
            3132
        ]
    }, 
    {
        "file": "/data/85.db", 
        "example": [
            4844
        ]
    }, 
    {
        "file": "/data/19.db", 
        "example": [
            1967
        ]
    }, 
    {
        "file": "/data/32.db", 
        "example": [
            1625
        ]
    }
]

gt_id_val = []

for s in gt_id:
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


with open("val_feature.pkl", "wb") as f:
    pickle.dump(primitives, f)
with open("val_label.pkl", "wb") as f:
    pickle.dump(ground, f)