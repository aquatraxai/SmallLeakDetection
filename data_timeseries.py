import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class TimeSeriesDataset():
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.max_consumption = np.max(self.df['Consumed'])
        self.no_of_samples_per_acc = {}  # no of days is no of days
        self.acc_number=0
        self.no_of_samples=0

    def plot_timeseries_per_account(self, df_per_account, plot_diff=False):

        account_number = df_per_account["AccountNumber"].iloc[0]
        if plot_diff=False:
            x_axis = df_per_account['Usagedate']
            x_axis = [str(i) for i in x_axis]
            labels = [i for i in x_axis]
            y_axis = df_per_account['Consumed']

            plt.figure(figsize=(20, 8))
            plt.plot(x_axis, y_axis)
            plt.xticks(x_axis, labels, rotation='vertical', fontsize=2)
            #plt.ylim(0, self.max_consumption)
            title = "Account number : " + str(account_number)
            plt.title(title)
            plt.show()
        else:


    def create_samples_per_account(self, df_per_account):

        print("Account number: {}".format(df_per_account['AccountNumber'].iloc[0]))

        df_per_account = df_per_account.sort_values(by='Usagedate')
        df_per_account = df_per_account.reset_index()

        df_per_account['Days'] = pd.to_datetime(df_per_account['Usagedate']).dt.date
        df_per_account['Time'] = pd.to_datetime(df_per_account['Usagedate']).dt.time
        self.acc_number += 1
        self.no_of_samples += len(df_per_account)
        self.no_of_samples_per_acc[self.acc_number] = len(df_per_account)

        self.plot_timeseries_per_account(df_per_account)

    def create_samples(self):
        account_grps = self.df.groupby('AccountNumber')
        for i in tqdm(range(len(np.unique(self.df['AccountNumber'])[:1]))):
            self.create_samples_per_account(account_grps.get_group(np.unique(self.df['AccountNumber'])[i]))



if __name__ == '__main__':
    csv_path = '/Users/laibamehnaz/Documents/Aquatrax/AMI_DATA.csv'
    dataset = TimeSeriesDataset(csv_path)
    dataset.create_samples()
    print("Total number of samples: {}".format(dataset.no_of_samples))
    print("Total number of samples per account: {}".format(dataset.no_of_samples_per_acc))