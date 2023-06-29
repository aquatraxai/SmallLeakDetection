import os
import argparse
import zipfile
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt1
import seaborn as sns
import matplotlib.cm as cm
import keras
import cv2
from sklearn.decomposition import PCA
# %matplotlib inline
# import K-Means
from sklearn.cluster import KMeans
# important metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from data import ImageDataset


class Trainer():

    def __init__(self, data_dir, processed_data_dir, output_dir, img_size, n):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.img_size = img_size
        self.reshaped_data = []
        self.processed_data_dir = processed_data_dir
        self.n = n
        self.pca_df = []

    def load_data(self):
        dataset = ImageDataset(self.data_dir, self.processed_data_dir, self.n)
        dataset.create_samples()
        print("Total number of samples: {}".format(dataset.no_of_samples))

    def preprocessing(self):
        data = []
        # label = []
        path = self.processed_data_dir
        IMG_SIZE = self.img_size

        for file in os.listdir(path):
            img = cv2.imread(path + file)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype('float32')
            data.append(img)
        data = np.array(data)

        data = data / 255.0
        reshaped_data = data.reshape(len(data), -1)
        self.reshaped_data = reshaped_data
        print("reshaped_data: {}".format(reshaped_data.shape))

        #Transform the data
        pca = PCA(2) # input: (n_samples, n_features) output: (n_samples, n_components)
        self.pca_df = pca.fit_transform(self.reshaped_data)
        print("pca df size: {}".format(self.pca_df.shape))
        print("pca df type: {}".format(type(self.pca_df)))

    def kmeans(self, n_clusters, n_init, max_iter, tol):

        kmeans_inertia = pd.DataFrame(data=[], index=(2,n_clusters), columns=['inertia'])
        colors = ['red', 'blue', 'aqua', 'teal', 'pink', 'yellow', 'tomato', 'olive', 'deeppink', 'hotpink']
        for num_clusters in range(2, n_clusters+1):
            kmeans = KMeans(n_clusters=num_clusters, n_init=n_init, max_iter=max_iter, tol=tol, random_state=0)
            #labels = kmeans.fit_predict(self.reshaped_data)
            label = kmeans.fit_predict(self.pca_df)
            print("labels shape : {}".format(label.shape))
            kmeans_inertia.loc[num_clusters] = kmeans.inertia_


            u_labels = np.unique(label)
            #print("u_labels: {}".format(u_labels))
            #print("self.pca_df[label == 0 , 0] : {}".format(len(self.pca_df[label == 0 , 0])))
            #print("self.pca_df[label == 0 , 1] : {}".format(len(self.pca_df[label == 0 , 1])))
            #print("self.pca_df[label == 1 , 0] : {}".format(len(self.pca_df[label == 1 , 0])))
            #print("self.pca_df[label == 1 , 1] : {}".format(len(self.pca_df[label == 1 , 1])))

            for i in u_labels:
              plt1.scatter(self.pca_df[label == i , 0] , self.pca_df[label == i , 1] , label = i)
              plt1.legend()
              plt1.savefig(self.output_dir + '_kmeans_' + str(num_clusters) + '.png')
          

        """
        #x_data = [i for i in range(self.img_size * self.img_size * 3)]
        for i in range(0, n_clusters): #num_clusters: 2,10
          plt1.scatter(x_data, kmeans.cluster_centers_[i], color=colors[i], alpha=0.2, s=70)
          #plt1.scatter(x_data, kmeans.cluster_centers_[1], color='blue', alpha=0.2, s=50)
          #plt.show()
        plt1.savefig(self.output_dir + '_kmeans_' + str(num_clusters) + '.png')
        """
        
        
        #clusters = kmeans.fit_predict(self.reshaped_data)
        #print(kmeans.cluster_centers_.shape)

        print(kmeans_inertia)
        kmeans_inertia = kmeans_inertia.sort_index()
        kmeans_inertia.plot(y='inertia', kind='line')
        #plt1.show()
        plt1.savefig(self.output_dir + '_kmeans_inertia_' + '.png')
        
        """
        x_data = [i for i in range(self.img_size * self.img_size * 3)]
        plt1.scatter(x_data, kmeans.cluster_centers_[0], color='red', alpha=0.2, s=70)
        plt1.scatter(x_data, kmeans.cluster_centers_[1], color='blue', alpha=0.2, s=50)
        # plt.show()
        plt1.savefig(self.output_dir + '_kmeans_' + '.png')
        """

if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    # Adding Arguments
    ap.add_argument("-data_dir", "--data_dir", required=True, type=str, help='unprocessed data directory')
    ap.add_argument("-img_size", "--img_size", required=True, type=int, help='img_size')
    ap.add_argument("-processed_data_dir", "--processed_data_dir", required=True, type=str, help='processed_data_dir')
    ap.add_argument("-output_dir", "--output_dir", required=True, type=str, help='output_dir directory')
    ap.add_argument("-n", "--n", required=True, type=int, help='number of samples in training dataset')
    ap.add_argument("-n_clusters", "--n_clusters", required=True, type=int, help='total number of clusters')
    ap.add_argument("-n_init", "--n_init", required=True, type=int, help='number of initializations')
    ap.add_argument("-max_iter", "--max_iter", required=True, type=int, help='number of iterations')
    ap.add_argument("-tol", "--tol", required=True, type=float, help='tolerance for convergence')

    # args = ap.parse_args()
    args = vars(ap.parse_args())

    for ii, item in enumerate(args):
        print(item + ': ' + str(args[item]))

    data_dir = args['data_dir']
    processed_data_dir = args['processed_data_dir']
    img_size = args['img_size']
    output_dir = args['output_dir']
    n = args['n']
    n_clusters = args['n_clusters']
    n_init = args['n_init']
    max_iter = args['max_iter']
    tol = args['tol']
    trainer = Trainer(data_dir, processed_data_dir, output_dir, img_size, n)
    # trainer.load_data()
    trainer.preprocessing()
    trainer.kmeans(n_clusters, n_init, max_iter, tol)


