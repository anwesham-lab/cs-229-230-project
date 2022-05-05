import pandas as pd
from pathlib import Path
import argparse

def label_first(filename):
    df = pd.read_csv(filename, header=None)
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    df.to_csv(filename, header=False, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add Arguments')
    parser.add_argument('--file', default='imdb_data_raw.csv', metavar='path', 
                        required=False, help='the path to workspace')
    args = parser.parse_args()
    label_first(filename=args.file)