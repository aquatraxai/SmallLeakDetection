import pandas as pd
import argparse
import numpy as np
from tqdm import tqdm
import seaborn
import matplotlib.pyplot as plt
import scipy
seaborn.set_palette(seaborn.color_palette())
import warnings; warnings.simplefilter('ignore')





def split(df, output_dir):

    leakMeters = ['4351180', '0800240', '4354160', '4951960', '3153360', '0250780', '2850660', '4403520', '1202220', '5751740', '3821030', '0751760', '1000320', '4300880']
    print("Number of leak meters: {}".format(len(leakMeters)))
    
    waterMeter = []
    for i in range(len(df)):
      wm = df.iloc[i]['AMIMeterNumber'][:7]
      waterMeter.append(wm)
    waterMeter = [ str(w) for w in waterMeter]
    df['waterMeter'] = waterMeter

    df_leaks = df[df['waterMeter'].isin(leakMeters)]
    df_leaks = df_leaks.sort_values(by='Usagedate')
    df_leaks = df_leaks.reset_index()
    print("Length of df with small leaks: {}".format(len(np.unique(df_leaks['waterMeter']))))
    df_leaks.to_csv(output_dir + "df_leaks.csv", index=False)

    df_non_leaks = df[~df['waterMeter'].isin(leakMeters)]
    df_non_leaks = df_non_leaks.sort_values(by='Usagedate')
    df_non_leaks = df_non_leaks.reset_index()
    print("Length of df without leaks: {}".format(len(np.unique(df_non_leaks['waterMeter']))))
    df_non_leaks.to_csv(output_dir + "df_non_leaks.csv", index=False)



if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    # Adding Arguments
    ap.add_argument("-data_dir", "--data_dir", required=True, type=str, help='unprocessed data directory')
    ap.add_argument("-n", "--n", required=True, type=int, help='number of samples in training dataset')
    ap.add_argument("-output_dir", "--output_dir", required=True, type=str, help='output_dir directory')

    args = vars(ap.parse_args())

    for ii, item in enumerate(args):
        print(item + ': ' + str(args[item]))

    data_dir = args['data_dir']
    n = args['n']    
    output_dir = args['output_dir']
    df = pd.read_csv(data_dir)
    split(df, output_dir)
    