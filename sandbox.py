from featurizer.database import Database
from featurizer.acollector import AttributesValCollector, total_node
import os
import numpy as np

op_folder = "/home/paulluh/Downloads/total"
attributes = ['COL_END', 'COL_START', 'NER', 'POS', 'POSITION', 'ROW_END', 'ROW_START', 'KEYWORD']
custume = [
    ('in_row', "select ir.source from IN_ROW ir, TEXT t where ir.target = t.id AND t.value = 'Birth place' ")
]

acollector = AttributesValCollector(op_folder, attributes, custume, 'president_mix')
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
    
    end += prim.shape[0]
    
    primitives[start:end, :] = prim
    node_id[start:end] = i

    start = end


