from featurizer.database import Database
from featurizer.acollector import AttributesValCollector, total_node
import os
import numpy as np
import sys
sys.path.append("/home/paulluh/reef/program_synthesis")
sys.path.append("/home/paulluh/reef")
from heuristic_generator import HeuristicGenerator

op_folder = "/home/paulluh/Downloads/total"
attributes = ['COL_END', 'COL_START', 'NER', 'POS', 'POSITION', 'ROW_END', 'ROW_START', 'KEYWORD']
custume = [
    ('in_row', "select ir.source from IN_ROW ir, TEXT t where ir.target = t.id AND t.value = 'Place of Birth' ")
]

acollector = AttributesValCollector(op_folder, attributes, custume, 'political')
attributes_dict = acollector.get_attributes()

number_nodes = total_node(op_folder)
primitives = None
node_id = np.zeros((number_nodes, ))


start = 0
end = 0

for f in os.listdir(op_folder):

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


gt_id = [891, 1487 ,2299 ,1456 ,1944 ,3297]
noise = [13766, 23704]

ground = []
for i in node_id:
    if i in gt_id:
        ground.append(1)
    else:
        ground.append(-1)
ground = np.array(ground)



val_primitive = np.zeros((4, prim.shape[1]))
val_ground = np.array([1, 1, -1, -1])

for i, tar in enumerate([891, 1487, 13766, 23704]):
    result = np.where(node_id == tar)[0]
    val_primitive[i, :] = primitives[result]


# print(result)


# print(np.max(primitives))

idx = None
hg = HeuristicGenerator(primitives, val_primitive, 
                        val_ground, ground, 
                        b=0.5)

for _ in range(0, 3):
    hg.run_synthesizer(max_cardinality=1, idx=idx, keep=3, model='dt')
    hg.run_verifier()
    print(hg.evaluate())
    hg.find_feedback()
    idx = hg.feedback_idx
    if idx == []:
        break
    # print(idx)

#  33         return self.val_accuracy, self.train_accuracy, self.val_coverage, self.train_coverage, calculate_f1(sel    f.train_marginals, self.b, self.train_ground)

                        
# if i == 3:
#     hg.run_synthesizer(max_cardinality=1, idx=idx, keep=3, model='dt')
# else:
#     hg.run_synthesizer(max_cardinality=1, idx=idx, keep=1, model='dt')
# hg.run_verifier()

# #Save evaluation metrics
# va,ta, vc, tc = hg.evaluate()


