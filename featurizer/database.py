import sqlite3
from sklearn import preprocessing
from collections import defaultdict
import numpy as np

class Database(object):
    def __init__(self, file_path, attributes, custome):

        print(file_path)
        
        conn = sqlite3.connect(file_path)
        cur = conn.cursor()

        self.collect_attribute = {}
        self.attributes = attributes

        for attr in attributes:

            try:
                cur.execute(
                    "select * from %s;" % attr
                )
                result = cur.fetchall()
                self.collect_attribute[attr] = result
            except:
                pass
    
        cur.execute(
            "select id from KIND;"
        )

        self.nodes = {}

        for (i,) in cur.fetchall():
            for a in attr:
                self.nodes[i] = dict(zip(attributes, ["None" for _ in range(len(attributes))]))


        for attr, result in self.collect_attribute.items():
            for (i, v) in result:
                self.nodes[i][attr] = v

        
        for (k, cmd) in custome:
            cur.execute(
                cmd
            )
            retrieve = set([i for (i, ) in cur.fetchall()])

            for (i, v) in self.nodes.items():
                if i in retrieve:
                    v[k] = '1'
                else:
                    v[k] = '0'

        cur.close()
        conn.close()
        

    def generate_primitive_matrix(self, attr_class):

        width = sum([len(l) for k, l in attr_class.items()])
        height = len(self.nodes.keys())
        
        primitive_matrix = np.zeros((height, width))
        vertex_id = []

        i = 0
        for node_id, attrs in self.nodes.items():
            vertex_id.append(node_id)
            j = 0
            for attr in attr_class.keys():
                lb = preprocessing.LabelBinarizer()
                lb.fit(attr_class[attr])
                one_hot_vector = lb.transform([attrs[attr]])[0]
                primitive_matrix[i][j:j+len(one_hot_vector)] = one_hot_vector
                j += len(one_hot_vector)
            i += 1
        
        return primitive_matrix, vertex_id


