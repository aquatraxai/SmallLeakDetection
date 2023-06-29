# SmallLeakDetection

To use this code, git clone this repository:
```
git clone https://github.com/aquatraxai/SmallLeakDetection.git
```
 
To split the dataset into two datasets containing leaks and non_leaks meters, use the dataSplit.py file as follows. Here, ```--n``` controls the number of samples to be used from the dataset, n == -1 uses the entire dataset.
```
python dataSplit.py --data_dir 'Usagedata.csv' \
  --output_dir 'output/' \
  --n -1  
```

To fill the missing data, i.e., fill in the consumption for the missing days for each meter, use the following code with either df_non_leaks.csv or df_leaks.csv.  
```
!python dataPreprocessing.py --data_dir '/df_non_leaks.csv' \
  --output_dir 'output/' \
  --n -1
```

To generate Fourier Transform graphs of meters, either on a weekly or a yearly basis, use the following code with any of the two datasets: df_non_leaks.csv or df_leaks.csv. Make sure to make two different output folders when using the two datasets. For example, as shown for df_leaks.csv dataset.
```
!python dataGeneration.py --data_dir 'df_leaks.csv' \
  --output_dir '/output/dukeLeaks/' \
  --n -1
```
