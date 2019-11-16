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
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import binarize
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

with open("test_feature.pkl", "rb") as f:
    primitives = pickle.load(f)

with open("test_label.pkl", "rb") as f:
    ground = pickle.load(f) 

with open("train_feature.pkl", "rb") as f:
    val_primitive = pickle.load(f)

with open("train_label.pkl", "rb") as f:
    val_ground = pickle.load(f)

with open("val_feature.pkl", "rb") as f:
    ml_val_primitive = pickle.load(f)

with open("val_label.pkl", "rb") as f:
    ml_val_ground = pickle.load(f)

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


clf = RandomForestRegressor(n_estimators=10, max_depth=2,
                             random_state=0)

# print(primitives, hg.vf.train_marginals)

clf.fit(primitives, hg.vf.train_marginals)
pred_y = clf.predict(ml_val_primitive)

bin_labels = binarize(pred_y.reshape(1, -1), threshold=0.5)
bin_labels[bin_labels == 0] = -1

print(np.unique(bin_labels, return_counts=True))

prec, recall, _, _ = precision_recall_fscore_support(ml_val_ground, bin_labels[0], average='binary')

f1 = 2*(prec * recall) / (prec + recall)

print("%f,%f,%f" % (prec, recall, f1))

    # print(idx)

#  33         return self.val_accuracy, self.train_accuracy, self.val_coverage, self.train_coverage, calculate_f1(sel    f.train_marginals, self.b, self.train_ground)

                        
# if i == 3:
#     hg.run_synthesizer(max_cardinality=1, idx=idx, keep=3, model='dt')
# else:
#     hg.run_synthesizer(max_cardinality=1, idx=idx, keep=1, model='dt')
# hg.run_verifier()

# #Save evaluation metrics
# va,ta, vc, tc = hg.evaluate()

