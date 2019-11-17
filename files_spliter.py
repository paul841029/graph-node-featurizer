from sklearn.model_selection import train_test_split
import argparse
import subprocess
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input")
parser.add_argument("--train")
parser.add_argument("--test")
parser.add_argument("--val")
args = parser.parse_args()

subprocess.call(["rm", "-rf", "%s" % args.train])
subprocess.call(["rm", "-rf", "%s" % args.test])
subprocess.call(["rm", "-rf", "%s" % args.val])

subprocess.call(["mkdir", "%s" % args.train])
subprocess.call(["mkdir", "%s" % args.test])
subprocess.call(["mkdir", "%s" % args.val])


all_db_files = filter(lambda x: '.db' in x, os.listdir(args.input))

train, test = train_test_split(all_db_files, train_size=0.2)
test, val = train_test_split(test, test_size=0.1/0.8)

for f in train:
    src = os.path.join(args.input, f)
    tar = os.path.join(args.train, f)
    subprocess.call(["cp", src, tar])

for f in test:
    src = os.path.join(args.input, f)
    tar = os.path.join(args.test, f)
    subprocess.call(["cp", src, tar])

for f in val:
    src = os.path.join(args.input, f)
    tar = os.path.join(args.val, f)
    subprocess.call(["cp", src, tar])

