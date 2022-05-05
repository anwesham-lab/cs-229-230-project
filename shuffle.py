import pandas as pd
from pathlib import Path
import argparse

def shuffle(filename):
    df = pd.read_csv(filename, header=None)
    ds = df.sample(frac=1)
    filename_new = filename[:-8] + 'shuffled.csv'
    ds.to_csv(filename_new, header=False, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add Arguments')
    parser.add_argument('--file', default='imdb_data_raw.csv', metavar='path', 
                        required=False, help='the path to workspace')
    args = parser.parse_args()
    shuffle(filename=args.file)