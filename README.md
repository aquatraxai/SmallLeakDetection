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
