import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class ImageDataset():
    def __init__(self, csv_path, output_dir, n):

        if n == -1:
          self.df = pd.read_csv(csv_path)
        else:
          self.nrows = n
          self.df = pd.read_csv(csv_path, nrows=self.nrows)
        self.max_consumption = np.max(self.df['Consumed'])
        self.no_of_samples = 0 #no of days is no of days
        self.output_dir = output_dir
        

    def create_samples_per_account_per_day(self, df):
        '''
        :param df: per day per account
        :return:
        '''

        self.no_of_samples += 1
        account_number = df['AccountNumber'].iloc[0]
        #print("Particular day for this account: ", df['Days'].iloc[0])
        x_axis = df['Time']
        x_axis = [i.strftime("%H:%M:%S") for i in x_axis]
        labels = [i[:2] for i in x_axis]
        y_axis = df['Consumed']


        plt.plot(x_axis, y_axis)
        plt.xticks(x_axis, labels, rotation='vertical')
        plt.ylim(0, self.max_consumption)
        title = "Account number : " + str(account_number) + ", Day: " + str(np.unique(df['Days'])[0])
        plt.title(title)
        #plt.show()
        plt.savefig(self.output_dir + '_image_no_' + str(self.no_of_samples) + '.png')


    def create_samples_per_account(self, df):
        '''
        :param df:single account several days
        :return: graphs of individual days of the given account showing the water consumption.
        '''

        #print("Creating samples for account number: {}".format(df['AccountNumber'].iloc[0]))
        df = df.sort_values(by='Usagedate')
        df = df.reset_index()

        df['Days'] = pd.to_datetime(df['Usagedate']).dt.date
        df['Time'] = pd.to_datetime(df['Usagedate']).dt.time

        no_of_days = np.unique(df['Days'])
        #print("Number of days for this account:  {}".format(len(no_of_days)))
        date_grps = df.groupby('Days')

        for day in no_of_days:
            self.create_samples_per_account_per_day(date_grps.get_group(day))

    def create_samples(self):

        account_grps = self.df.groupby('AccountNumber')

        for i in tqdm(range(len(np.unique(self.df['AccountNumber'])))):
            self.create_samples_per_account(account_grps.get_group(np.unique(self.df['AccountNumber'])[i]))




