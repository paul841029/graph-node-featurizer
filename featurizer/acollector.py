from multiprocessing import Pool
import sys
import sqlite3
from collections import defaultdict
sys.path.append("/home/paulluh/cosmos_ssd_3/hera-db-lib")
from tools import db_file_path_iterator
from batch_process import run_on_all_files_retrieve
import pickle

ATTRIBUTES = None

def single_db_op(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    rval_dict = defaultdict(set)
    global ATTRIBUTES
    for attr in ATTRIBUTES:
        try:
            cur.execute(
                """
                select value from %s;
                """ % attr

            )
            for (val,) in cur.fetchall():
                rval_dict[attr].add(str(val))
        except:
            pass   

    
    cur.close()
    conn.close()

    return rval_dict


class AttributesValCollector(object):
    def __init__(self, db_folder, attributes, custome, db_name):

        try:
            with open("%s.pkl" % db_name, "rb") as f:
                self.final_dict = pickle.load(f)
            raise IOError
        
        except IOError:

            global ATTRIBUTES
            ATTRIBUTES = attributes
        
            pool = Pool(processes=4)
            rval_dicts = pool.map(single_db_op, db_file_path_iterator(db_folder))

            self.final_dict = defaultdict(set)

            for dic in rval_dicts:
                for (k, v) in dic.items():
                    self.final_dict[k] |= v
            
            for k in attributes:
                self.final_dict[k] = list(self.final_dict[k])\
                     + ['None']
            
            for (k, _) in custome:
                self.final_dict[k] = ['0', '1']
        
            with open("%s.pkl" % db_name, "wb") as f:
                pickle.dump(self.final_dict, f)
    
    def get_attributes(self):
        return self.final_dict

def total_node(db_folder):
    rval_dict = run_on_all_files_retrieve(db_folder, "select id from KIND;")
    return sum([
        len(v) for (_, v) in rval_dict.items()
    ])
    
    



