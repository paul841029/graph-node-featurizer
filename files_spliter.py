from sklearn.model_selection import train_test_split
import argparse
import subprocess
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input")
parser.add_argument("--train")
parser.add_argument("--test")
args = parser.parse_args()

subprocess.call(["rm", "%s/*.db" % args.train])
subprocess.call(["rm", "%s/*.db" % args.test])

all_db_files = list(os.listdir(args.input))

train, test = train_test_split(all_db_files, train_size=0.2)



