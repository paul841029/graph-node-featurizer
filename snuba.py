# val_primitive = np.zeros((4, prim.shape[1]))
# val_ground = np.array([1, 1, -1, -1])

# for i, tar in enumerate([891, 1487, 13766, 23704]):
#     result = np.where(node_id == tar)[0]
#     val_primitive[i, :] = primitives[result]


# print(result)
import time

# print(np.max(primitives))
import sys
sys.path.append("/luh/snuba_experiment/reef/program_synthesis")
sys.path.append("/luh/snuba_experiment/reef")

import pickle
from heuristic_generator import HeuristicGenerator
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import binarize
# from sklearn.metrics import f1_score
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import precision_recall_fscore_support
import numpy as np

import argparse
import time


parser = argparse.ArgumentParser()
parser.add_argument("--dataset")
parser.add_argument("--gt_level")
parser.add_argument("--example", type=int)
# parser.add_argument("--threshold", type=float)
args = parser.parse_args()

with open("train_feature_%s.pkl" % args.dataset, "rb") as f:
    train_primitive = pickle.load(f)

with open("train_label_%s.pkl" % args.dataset, "rb") as f:
    train_label = pickle.load(f)

with open("test_feature_%s.pkl" % args.dataset, "rb") as f:
    test_primitive = pickle.load(f)

with open("test_label_%s.pkl" % args.dataset, "rb") as f:
    test_label = pickle.load(f)

num_pos_example = -1
if args.example == -1:
    with open("num_pos_example_%s.pkl" % args.dataset, "rb") as f:
        num_pos_example = pickle.load(f) 

# with open("val_feature_%s.pkl" % args.dataset, "rb") as f:
#     ml_val_primitive = pickle.load(f)

# with open("val_label.pkl", "rb") as f:
#     ml_val_ground = pickle.load(f)
start_time = time.time()
idx = None
hg = HeuristicGenerator(test_primitive, train_primitive, 
                        train_label, train_ground=test_label, 
                        b=0.5)

for _ in range(0, 10):
    hg.run_synthesizer(max_cardinality=1, idx=idx, keep=3, model='dt')
    hg.run_verifier()
    hg.find_feedback()
    idx = hg.feedback_idx
    if idx == []:
        break

unique, counts = np.unique(train_label, return_counts=True)
num_count = dict(zip(unique, counts))

p_example = args.example if args.example is not -1 else num_pos_example

for t in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:

    _, _, _, _, (prec, recall, f1) = hg.evaluate(t)

    with open("results_log.csv", "a") as f:
        f.write(
            "\n%s,%f,%f,%f,%d,%d,%f,%s,%f" % (args.dataset, prec, recall, f1, p_example, num_count[-1], t, args.gt_level, time.time() - start_time)
            )

# clf = RandomForestRegressor(n_estimators=10, max_depth=2,
#                              random_state=0)

# # print(primitives, hg.vf.train_marginals)

# clf.fit(primitives, hg.vf.train_marginals)
# pred_y = clf.predict(ml_val_primitive)

# bin_labels = binarize(pred_y.reshape(1, -1), threshold=0.5)
# bin_labels[bin_labels == 0] = -1

# print(np.unique(bin_labels, return_counts=True))

# prec, recall, _, _ = precision_recall_fscore_support(ml_val_ground, bin_labels[0], average='binary')

# f1 = 2*(prec * recall) / (prec + recall)



    # print(idx)

#  33         return self.val_accuracy, self.train_accuracy, self.val_coverage, self.train_coverage, calculate_f1(sel    f.train_marginals, self.b, self.train_ground)

                        
# if i == 3:
#     hg.run_synthesizer(max_cardinality=1, idx=idx, keep=3, model='dt')
# else:
#     hg.run_synthesizer(max_cardinality=1, idx=idx, keep=1, model='dt')
# hg.run_verifier()

# #Save evaluation metrics
# va,ta, vc, tc = hg.evaluate()

