import pandas as pd
import argparse
import numpy as np
from pandas._libs.tslibs.parsing import try_parse_date_and_time
from tqdm import tqdm
import seaborn
import matplotlib.pyplot as plt
import scipy
seaborn.set_palette(seaborn.color_palette())
import warnings; warnings.simplefilter('ignore')

class DataAugment():
    def __init__(self, csv_path, n, output_dir):

        if n == -1:
          self.df = pd.read_csv(csv_path)
        else:
          self.nrows = n
          self.df = pd.read_csv(csv_path, nrows=self.nrows)

        self.aug_df = []
        self.output_dir = output_dir

    def perWeekData(self):
        
      return self

    def create_samples_per_account_per_day(self, df, day):
        '''
        :param df: per day per account
        :return:
        '''

        #self.no_of_samples += 1
        account_number = df['AccountNumber'].iloc[0]
        #print("Particular day for this account: ", df['Days'].iloc[0])
        
        df['FFT_W'] = [i/len(df) for i in range(len(df))]
        df['FFT_Period'] = 1/df['FFT_W']
        df['FFT'] = scipy.fft.fft(df['Value'])
        df['FFTA'] = np.abs(df['FFT'])

        
        plt.title(f'Discrete Fourier Transform on Water Usage of account {account_number} for day {day}')
        g = seaborn.lineplot(data=df,x="FFT_Period",y="FFTA",color='b',size=0.01)
        g.set(xlim=(0, 30))
        g.set_xlabel(f'Signal Period (h)')
        g.set_ylabel(f'Signal Amplitude')
        legend = g.legend()
        g.grid()
        legend.remove()

        fig = g.get_figure()
        fig.savefig(self.output_dir + "fourier_" +  str(account_number) + "_" + str(day) +"_.png") 


    def plot_graph(self, df, weekly=True):
        '''
        :param df: per week per account
        :return:
        '''
        account_number = df['AccountNumber'].iloc[0]
        if weekly == True:
          week_number = df['WeekNumber'].iloc[0]
           
        #augment
        base = 100
        meter_reading = []
        reading = base
        meter_reading.append(reading)
        for i in range(len(df)-1):
            reading = reading + df.iloc[i]['Consumed'] 
            meter_reading.append(reading)
        assert len(meter_reading) == len(df)
        df['MeterReading'] = meter_reading
        small_leak = 0.4
        df['small_leak'] = small_leak

        potential_reading = base + small_leak
        potential_meter_reading = []
        potential_meter_reading.append(potential_reading)
        for i in range(len(df)-1):
            potential_reading = potential_reading + df.iloc[i]['small_leak'] + df.iloc[i]['Consumed'] 
            potential_meter_reading.append(round(potential_reading, 2))
        
        assert len(potential_meter_reading) == len(df)
        df['PotentialReading'] = potential_meter_reading

        new_consumption = []
        new_cons = small_leak
        new_consumption.append(new_cons)
        for i in range(len(df)-1):
            new_cons = df.iloc[i+1]['PotentialReading'] -  df.iloc[i]['PotentialReading']
            new_consumption.append(round(new_cons,2))

        assert len(new_consumption) == len(df)
        df['NewConsumption'] = new_consumption
        
        leaking_meter_reading = [int(df.iloc[0]['PotentialReading'])]
        for i in range(len(df)-1):
          leaking_meter_reading.append(int(df.iloc[i+1]['PotentialReading']))

        assert len(leaking_meter_reading) == len(df)
        df['LeakingMeterReading'] = leaking_meter_reading
        df.to_csv(self.output_dir + "augmented_data.csv")

        #plotting augmented 
        
        df['FFT_W'] = [i/len(df) for i in range(len(df))]
        df['FFT_Period'] = 1/df['FFT_W']
        #df['FFT'] = scipy.fft.fft(df['PotentialReading'].values)
        df['FFT'] = scipy.fft.fft(df['NewConsumption'].values)
        df['FFTA'] = np.abs(df['FFT'])

        if weekly == True:
          plt.title(f'Discrete Fourier Transform on Augmented Water Usage of account {account_number} for week {week_number}')
        else:
          plt.title(f'Discrete Fourier Transform on Augmented Water Usage of account {account_number}')
  
        g = seaborn.lineplot(data=df,x="FFT_Period",y="FFTA",color='b',size=0.01)
        g.set(xlim=(0, 30))
        g.set_xlabel(f'Signal Period (h)')
        g.set_ylabel(f'Signal Amplitude')
        legend = g.legend()
        g.grid()
        legend.remove()

        fig = g.get_figure()
        if weekly == True:
          fig.savefig(self.output_dir + "augmented_fourier_" +  str(account_number) + "_" + str(week_number) +"_.png")
        else:
          fig.savefig(self.output_dir + "augmented_fourier_" +  str(account_number) +"_.png")



        #plotting original
        df['FFT_W'] = [i/len(df) for i in range(len(df))]
        df['FFT_Period'] = 1/df['FFT_W']
        #df['FFT'] = scipy.fft.fft(df['MeterReading'])
        df['FFT'] = scipy.fft.fft(df['Consumed'])
        df['FFTA'] = np.abs(df['FFT'])

        if weekly == True:
          plt.title(f'Discrete Fourier Transform on Water Usage of account {account_number} for week {week_number}')
        else:
          plt.title(f'Discrete Fourier Transform on Water Usage of account {account_number}')

        g = seaborn.lineplot(data=df,x="FFT_Period",y="FFTA",color='b',size=0.01)
        g.set(xlim=(0, 30))
        g.set_xlabel(f'Signal Period (h)')
        g.set_ylabel(f'Signal Amplitude')
        legend = g.legend()
        g.grid()
        legend.remove()

        fig = g.get_figure()
        if weekly == True:
          fig.savefig(self.output_dir + "fourier_" +  str(account_number) + "_" + str(week_number) +"_.png") 
        else:
          fig.savefig(self.output_dir + "fourier_" +  str(account_number) +"_.png") 

    def plot_weekly_data(self, df):

        num_days = df['Days'].unique()
        
        if len(num_days) <7:
          print("Not a complete week.")

        else:
          print("Complete week.")
          self.plot_graph(df)
      
    def plot_yearly_data(self, df):
        
        self.plot_graph(df, weekly=False)

    def create_samples_per_account(self, df, weekly= True):
        '''
        Each account has only one meter.
        :param df: single account several days
        :return:
        '''
        print("*********************************************")
        print("Account : {}".format(df['AccountNumber'].iloc[0]))
        # print("Unique meters: {}".format(np.unique(df['waterMeter'])))
        # print(type(df['AMIMeterNumber'].iloc[0]))

        df = df.sort_values(by='Usagedate')
        df = df.reset_index()

        df['Days'] = pd.to_datetime(df['Usagedate']).dt.date
        df['Time'] = pd.to_datetime(df['Usagedate']).dt.time

        df = df.sort_values(by='Days')
        no_of_days = np.unique(df['Days'])
        date_grps = df.groupby('Days')

        # checking if consecutive days are present to get weekly data
        day_diff = []
        prev_day = no_of_days[0]
        for day in no_of_days:
            day_diff.append((day - prev_day).days)
            prev_day = day

        # print(day_diff)
        print("This account has {} inconsistencies".format(sum(diff > 1 for diff in day_diff)))

        if weekly != True:
            self.plot_yearly_data(df)

        # Monday: 0, Tuesday: 1, Wednesday:2....
        weeknumbers = []
        for i in range(len(df)):
            weeknumber = df['Days'].iloc[i].isocalendar().week
            weeknumbers.append(weeknumber)

        df['WeekNumber'] = weeknumbers
        week_grps = df.groupby('WeekNumber')
        for i in range(len(np.unique(weeknumbers))):
            self.plot_weekly_data(week_grps.get_group(np.unique(weeknumbers)[i]))

    def createSmallLeaksData(self):

      print("columns : {}".format(self.df.columns))
      account_grps = self.df.groupby('AccountNumber')
      for i in range(len(np.unique(self.df['AccountNumber']))):
          self.create_samples_per_account(account_grps.get_group(np.unique(self.df['AccountNumber'])[i]), weekly=False)
      
      return
      

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
    data = DataAugment(data_dir, n, output_dir)
    data.createSmallLeaksData()





