import pandas as pd
import numpy as np
import argparse

def clean(filename, prefix):
    raw = pd.read_csv(filename)
    raw.drop("PhraseId", axis=1, inplace=True)
    raw.drop("SentenceId", axis=1, inplace=True)
    raw.rename(columns={'Phrase': 'input', 'Sentiment': 'label'}, inplace=True)
    raw.loc[raw["label"] == 1, "label"] = 0
    raw.loc[raw["label"] == 3, "label"] = 1
    raw.loc[raw["label"] == 4, "label"] = 1
    filename_new = prefix + 'cleaned.csv'
    raw.to_csv(filename_new, index=False)
    return raw

def sample(df, count):
    df_pos = df.drop(df.loc[df['label']==0].index, inplace=False)
    df_neg = df.drop(df.loc[df['label']==1].index, inplace=False)
    # Go through and establish training pieces of positive based on counts
    pos_eval = df_pos.sample(frac=((count/16)/len(df_pos)))
    df_pos = df_pos.drop(pos_eval.index)
    pos_test = df_pos.sample(frac=((count/16)/len(df_pos)))
    df_pos = df_pos.drop(pos_test.index)
    pos_train = df_pos.sample(frac=((count/2)/len(df_pos)))
    # Do the same for the negative
    neg_eval = df_neg.sample(frac=((count/16)/len(df_neg)))
    df_neg = df_neg.drop(neg_eval.index)
    neg_test = df_neg.sample(frac=((count/16)/len(df_neg)))
    df_neg = df_neg.drop(neg_test.index)
    neg_train = df_neg.sample(frac=((count/2)/len(df_neg)))
    return [pos_train, neg_train, pos_eval, neg_eval, pos_test, neg_test]

def merge(sampled, prefix):
    train_dfs, eval_dfs, test_dfs = sampled[:2], sampled[2:4], sampled[4:]
    train = pd.concat(train_dfs).sample(frac=1)
    eval = pd.concat(eval_dfs).sample(frac=1)
    test = pd.concat(test_dfs).sample(frac=1)
    train.to_csv(prefix + 'train.csv', index=False)
    eval.to_csv(prefix + 'eval.csv', index=False)
    test.to_csv(prefix + 'test.csv', index=False)
    return [train, eval, test]

def main(filename):
    prefix = 'rotten_tomatoes_'
    cleaned = clean(filename, prefix)
    sampled = sample(cleaned, 40000)
    merged = merge(sampled, prefix)
    print("Length of Training Dataset: ", len(merged[0]))
    print("Length of Evaluation Dataset: ", len(merged[1]))
    print("Length of Testing Dataset: ", len(merged[2]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add Arguments')
    parser.add_argument('--file', default='raw.csv', metavar='path', 
                        required=False, help='the path to workspace')
    args = parser.parse_args()
    main(filename=args.file)