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
from datetime import datetime
import math

class Preprocessing():
  
  def __init__(self, csv_path, n, output_dir):
    if n == -1:
      self.df = pd.read_csv(csv_path)
    else:
      self.nrows = n
      self.df = pd.read_csv(csv_path, nrows=self.nrows)

    self.aug_df = []
    self.output_dir = output_dir
    self.imputed_df = []
    self.i=0 # counter for appending the df into the imputed df

  def save_imputed_df(self):
      print("Final imputed df size: {}".format(self.imputed_df.shape))
      self.imputed_df = self.imputed_df.append(self.df) 
      self.imputed_df = self.imputed_df.sort_values(['AccountNumber','Days', 'Time'])
      self.imputed_df.to_csv(self.output_dir + "df_non_leaks_imputed.csv")

  def avg_per_hour_consumption(self, df):
    '''
    returns: avg_per_hour_cons is a list with 24 values
    '''
    df['Hour'] = pd.to_datetime(df['Usagedate']).dt.hour
    no_of_hours = np.unique(df['Hour'])
    hour_grps = df.groupby('Hour')
    avg_per_hour_cons = []
    for hour in no_of_hours:
      avg_per_hour = math.ceil(hour_grps.get_group(hour)['Consumed'].mean())
      avg_per_hour_cons.append(avg_per_hour)
      #print("Avg consumption for hour {} is {}".format(hour, avg_per_hour))
    return avg_per_hour_cons

  def preprocess_per_month(self, df):
    """
    for col in df.columns:
      print("Colummns: {}".format(col))
      print(df.iloc[:10][col])
      print("\n")
    """
    start = df.iloc[0]['Usagedate']
    end = df.iloc[len(df)-1]['Usagedate']
    missing_days = [ dt for dt in pd.date_range(start=start, end=end).difference(df.Usagedate)]
    print("Missing days: {} for month: {}".format(len(missing_days), df.iloc[0]['Month']))    

    avg_per_hour_cons = self.avg_per_hour_consumption(df) # avg consumption per hour per month will be same for all missing days

    account_number = [df.iloc[0]['AccountNumber']] * 24
    segment = [df.iloc[0]['Segment']] * 24
    metersize = [df.iloc[0]['MeterSize']] * 24
    ami_meter_number = [df.iloc[0]['AMIMeterNumber']] * 24
    water_meter =  [df.iloc[0]['waterMeter']] * 24
    #print("Type of UsageDate sample: {}".format(type(df.iloc[0]['Usagedate']))) # string version of datetime

    
    for missing_day in missing_days:
      if type(missing_day) != type(df.iloc[0]['Days']):
        missing_day = missing_day.date()
        #print(missing_day)
        #print(df.iloc[0]['Days'])
        #print(type(df.iloc[0]['Days']))
        #print("After typecasting: {}".format(type(missing_day) == type(df.iloc[0]['Days'])))
      usage_date = []
      days = [missing_day] * 24
      time = np.unique(df['Time'])
      month = [missing_day.month] * 24

      ## converting str to usageDate
      for t in time:
        dt_str = str(missing_day.strftime("%y/%m/%d")) +" "+ t.strftime("%H:%M:%S")
        usage_date.append(dt_str)
      assert len(usage_date) == 24

      #adding missing dates for this month into a dataframe
      df = pd.DataFrame(account_number, columns=['AccountNumber'])
      df['Segment'] = segment
      df['MeterSize'] = metersize
      df['AMIMeterNumber'] = ami_meter_number
      df['Usagedate'] = usage_date
      df['Consumed'] = avg_per_hour_cons
      df['waterMeter'] = water_meter
      df['Days'] = days
      df['Time'] = time
      df['Month'] = month
      print("Dataframe created for missing dates with shape: {}".format(df.shape))

      if self.i==0:
        self.imputed_df = df
        self.i+=1
      else:
        self.imputed_df = self.imputed_df.append(df)
        #print("After appending the imputed df shape: {}".format(self.imputed_df.shape))

  def preprocess_per_account(self, df):

    print("*********************************************")
    print("Account : {}".format(df['AccountNumber'].iloc[0]))
    df = df.sort_values(by='Usagedate')
    df = df.reset_index()

    df['Days'] = pd.to_datetime(df['Usagedate']).dt.date
    df['Time'] = pd.to_datetime(df['Usagedate']).dt.time
    no_of_days = np.unique(df['Days'])
    df['Month'] = pd.to_datetime(df['Usagedate']).dt.month
    no_of_months = np.unique(df['Month'])
  
    month_grps = df.groupby('Month')
    #checking if consecutive days are present to get weekly data
    day_diff=[]
    prev_day = no_of_days[0]
    for day in no_of_days:
        day_diff.append((day-prev_day).days)
        prev_day = day
    
    #print(day_diff)
    print("This account has {} inconsistencies".format(sum(diff > 1 for diff in day_diff)))
    for month in no_of_months:
            self.preprocess_per_month(month_grps.get_group(month))


  def preprocess(self):

    account_grps = self.df.groupby('AccountNumber')
    for i in range(len(np.unique(self.df['AccountNumber'])[:1])):
      self.preprocess_per_account(account_grps.get_group(np.unique(self.df['AccountNumber'])[i]))    




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
    prep = Preprocessing(data_dir, n, output_dir)
    prep.preprocess()
    prep.save_imputed_df()






