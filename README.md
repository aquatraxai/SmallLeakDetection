
## 1. Installation
To use this code, git clone this repository:
```
git clone https://github.com/aquatraxai/SmallLeakDetection.git
```

## 2. Preprocessing
To split the dataset into two datasets containing leaks and non_leaks meters, use the dataSplit.py file as follows. Here, ```--n``` controls the number of samples to be used from the dataset, n == -1 uses the entire dataset. The code generates two files: df_non_leaks.csv and df_leaks.csv in the output folder.
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

## 3. Modeling
To generate Fourier Transform graphs of meters, either on a weekly or a yearly basis, use the following code with any of the two datasets: df_non_leaks.csv or df_leaks.csv. Make sure to make two different output folders when using the two datasets. For example, as shown for df_leaks.csv dataset.
```
!python dataGeneration.py --data_dir 'df_leaks.csv' \
  --output_dir '/output/dukeLeaks/' \
  --n -1
```

To train a k-means clustering model on the dataset, use the following code. ```--data_dir``` contains the input data with leaky or non-leaky meters. ```--img_size``` decides the number of features to be created per image(graph showing consumption. ofwater per day per account).```--processed_data_dir``` is the directory where the images are saved for further use in this file for training the model. ```--output_dir``` is the directory where the ouput of k-means clustering algorithm is saved(clusters in the dataset). ```--n``` controls the number of samples of the dataset to be used for training. Keep the rest of the arguments as it is for stable training. 
```
!python train.py --data_dir '/Usagedata.csv' \
  --img_size 32 \
  --processed_data_dir '/img_data/' \
  --output_dir '/output/' \
  --n -1 \
  --n_clusters 10 \
  --n_init 10 \
  --max_iter 100 \
  --tol 0.0001
```

