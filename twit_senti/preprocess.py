import pandas as pd
import numpy as np
import argparse

def make_binary(df):
    df_positive = df[df["label"] == "Positive"]
    df_negative = df[df["label"] == "Negative"]
    df_binary = pd.concat([df_positive, df_negative])
    return df_binary

def make_numeric(df):
    df.loc[df["label"] == "Positive", "label"] = 1
    df.loc[df["label"] == "Negative", "label"] = 0
    return df

def split_data(prefix, df):
    train, eval, test = np.split(df.sample(frac=1, random_state=42), [int(len(df) - 4000), int(len(df) - 2000)])
    train.to_csv(prefix + 'train.csv', index=False)
    eval.to_csv(prefix + 'eval.csv', index=False)
    test.to_csv(prefix + 'test.csv', index=False)
    return train, eval, test

def main(train_file, test_file):
    prefix = 'twitsenti_'
    df1 = pd.read_csv(train_file, names=["id", "entity", "label", "input"], usecols=["input", "label"])
    df2 = pd.read_csv(test_file, names=["id", "entity", "label", "input"], usecols=["input", "label"])
    df = pd.concat([df1, df2]).sample(frac=1)
    df_binary = make_binary(df)
    df_processed = make_numeric(df_binary)
    df_processed.to_csv(prefix + 'processed.csv', index=False)
    train, eval, test = split_data(prefix, df_processed)
    print("Length of Training Dataset: ", len(train))
    print("Length of Evaluation Dataset: ", len(eval))
    print("Length of Testing Dataset: ", len(test))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add Arguments')
    parser.add_argument('--trainfile', default='twitter_training.csv', metavar='path', 
                        required=False, help='path to unprocessed train file')
    parser.add_argument('--testfile', default='twitter_validation.csv', metavar='path', 
                        required=False, help='path to unprocessed test file')
    args = parser.parse_args()
    main(train_file=args.trainfile, test_file=args.testfile)