# raise RuntimeError("ARE YOU SURE?")
from sklearn.model_selection import train_test_split
import argparse
import subprocess
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument("--input")
parser.add_argument("--train")
parser.add_argument("--test")
parser.add_argument("--data")
args = parser.parse_args()

# subprocess.call(["rm", "-rf", "%s" % args.train])
# subprocess.call(["rm", "-rf", "%s" % args.test])
# subprocess.call(["rm", "-rf", "%s" % args.val])

subprocess.call(["mkdir", "-p", "%s" % args.train])
subprocess.call(["mkdir", "-p", "%s" % args.test])
# subprocess.call(["mkdir", "%s" % args.val])


all_db_files = filter(lambda x: '.db' in x, os.listdir(args.input))

train, test = train_test_split(all_db_files, train_size=0.2)
# train = all_db_files[:3]
# test = all_db_files[3:6]
# val = all_db_files[6:]

print(len(train))
print(len(test))
# print(len(val))

for f in train:
    src = os.path.join(args.input, f)
    tar = os.path.join(args.train, f)
    subprocess.call(["cp", src, tar])

for f in test:
    src = os.path.join(args.input, f)
    tar = os.path.join(args.test, f)
    subprocess.call(["cp", src, tar])

with open("%s_split.json" % args.data, "w") as f:
    json.dump({
        "dataset": args.data,
        "train": train,
        "test": test        
    }, f, indent=4)
# for f in val:
#     src = os.path.join(args.input, f)
#     tar = os.path.join(args.val, f)
#     subprocess.call(["cp", src, tar])

