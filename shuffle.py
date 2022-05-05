import pandas as pd
import numpy as np
import argparse

def shuffle(filename, prefix):
    df = pd.read_csv(filename, header=None, names=['label', 'message'])
    ds = df.sample(frac=1, random_state=30)
    filename_new = prefix + 'shuffled.csv'
    ds['message'] = ds['message'].str.replace('"', '')
    ds.to_csv(filename_new, header=False, index=False)
    return ds

def split_data(prefix, ds):
    train, dev, test = np.split(ds.sample(frac=1, random_state=42), [int(.8*len(ds)), int(.9*len(ds))])
    train.to_csv(prefix + 'train.csv', header=False, index=False)
    dev.to_csv(prefix + 'dev.csv', header=False, index=False)
    test.to_csv(prefix + 'test.csv', header=False, index=False)

def main(filename, split):
    prefix = filename[:-7]
    ds = shuffle(filename, prefix)
    if split:
        split_data(prefix, ds)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add Arguments')
    parser.add_argument('--file', default='imdb_data_raw.csv', metavar='path', 
                        required=False, help='the path to workspace')
    parser.add_argument('--split', default=True, type=bool, 
                        required=False, help='the path to workspace')
    args = parser.parse_args()
    main(filename=args.file, split=args.split)