import numpy as np
import re
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
#from imdb import IMDb
#from requests import get
#from bs4 import BeautifulSoup
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def run_pca(df_standard, df):
    print("\n\n----------------------Principal Component Analysis----------------------\n\n")
    pca = PCA(n_components=3)
    df_pca = pca.fit_transform(df_standard)
    print("Explained Variance Ratio: ",pca.explained_variance_ratio_)

    df_standard['pca_one'] = df_pca[:, 0]
    df_standard['pca_two'] = df_pca[:, 1]
    df_standard['pca_three'] = df_pca[:, 2]

    plt.scatter(df_standard['pca_one'], df_standard['pca_two'], color=['orange', 'cyan', 'brown'], cmap='viridis')
    plt.show()
