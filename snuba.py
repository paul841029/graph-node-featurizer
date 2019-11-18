# val_primitive = np.zeros((4, prim.shape[1]))
# val_ground = np.array([1, 1, -1, -1])

# for i, tar in enumerate([891, 1487, 13766, 23704]):
#     result = np.where(node_id == tar)[0]
#     val_primitive[i, :] = primitives[result]


# print(result)


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

parser = argparse.ArgumentParser()
parser.add_argument("--dataset")
parser.add_argument("--gt_level")
parser.add_argument("--threshold", type=float)
args = parser.parse_args()

with open("train_feature_%s.pkl" % args.dataset, "rb") as f:
    train_primitive = pickle.load(f)

with open("train_label_%s.pkl" % args.dataset, "rb") as f:
    train_label = pickle.load(f)

with open("test_feature_%s.pkl" % args.dataset, "rb") as f:
    test_primitive = pickle.load(f)

with open("test_label_%s.pkl" % args.dataset, "rb") as f:
    test_label = pickle.load(f) 

# with open("val_feature_%s.pkl" % args.dataset, "rb") as f:
#     ml_val_primitive = pickle.load(f)

# with open("val_label.pkl", "rb") as f:
#     ml_val_ground = pickle.load(f)

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
_, _, _, _, (prec, recall, f1) = hg.evaluate(args.threshold)

unique, counts = np.unique(train_label, return_counts=True)
num_count = dict(zip(unique, counts))

with open("results_log.csv", "a") as f:
    f.write(
        "\n%s,%f,%f,%f,%d,%d,%f" % (args.dataset, prec, recall, f1, num_count[1], num_count[-1], args.threshold)
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

