import pandas as pd
import numpy as np
import argparse

def clean(train_file, test_file):
    orig_train = pd.read_csv(train_file)
    orig_test = pd.read_csv(test_file)
    orig_train.drop("id", axis=1, inplace=True)
    orig_test.drop("id", axis=1, inplace=True)
    return orig_train.sample(frac=1), orig_test

def make_eval(orig_test):
    eval = orig_test.sample(frac=0.5)
    test = orig_test.drop(eval.index)
    test = test.sample(frac=1)
    return eval, test

def writeout(prefix, train, eval, test):
    train.to_csv(prefix + 'train.csv', index=False)
    eval.to_csv(prefix + 'eval.csv', index=False)
    test.to_csv(prefix + 'test.csv', index=False)

def main(train_file, test_file):
    prefix = 'twitprod_'
    train, orig_test = clean(train_file, test_file)
    eval, test = make_eval(orig_test)
    writeout(prefix, train, eval, test)
    print("Length of Training Dataset: ", len(train))
    print("Length of Evaluation Dataset: ", len(eval))
    print("Length of Testing Dataset: ", len(test))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add Arguments')
    parser.add_argument('--trainfile', default='orig_train.csv', metavar='path', 
                        required=False, help='path to unprocessed train file')
    parser.add_argument('--testfile', default='orig_test.csv', metavar='path', 
                        required=False, help='path to unprocessed test file')
    args = parser.parse_args()
    main(train_file=args.trainfile, test_file=args.testfile)