import pandas as pd
import numpy as np
import argparse

def shuffle(filename):
    df = pd.read_csv(filename, header=None, names=['message', 'label'])
    ds = df.sample(frac=1, random_state=30)
    prefix = filename[:-7]
    filename_new = prefix + 'shuffled.csv'
    ds['message'] = ds['message'].str.replace('"', '')
    ds.to_csv(filename_new, header=False, index=False)

    train, dev, test = np.split(ds.sample(frac=1, random_state=42), [int(.8*len(df)), int(.9*len(df))])
    train.to_csv(prefix + 'train.csv', header=False, index=False)
    dev.to_csv(prefix + 'dev.csv', header=False, index=False)
    test.to_csv(prefix + 'test.csv', header=False, index=False)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add Arguments')
    parser.add_argument('--file', default='imdb_data_raw.csv', metavar='path', 
                        required=False, help='the path to workspace')
    args = parser.parse_args()
    shuffle(filename=args.file)