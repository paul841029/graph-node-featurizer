import sys
sys.path.append("/home/paulluh/reef/program_synthesis")
sys.path.append("/home/paulluh/reef")
from heuristic_generator import HeuristicGenerator

hg = HeuristicGenerator(train_primitive_matrix, val_primitive_matrix, val_ground, train_ground, b=0.5)
hg.run_synthesizer(max_cardinality=1, idx=None, keep=3, model='dt')
hg.run_verifier()
va,ta, vc, tc = hg.evaluate()