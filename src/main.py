import pandas as pd
from sklearn.preprocessing import StandardScaler
from pca import run_pca
from knn import run_knn
from random_forest import run_random_forest
from logistic import run_logistic_regression
from xgboostModel import run_xgboost_cornell
from xgboostModel import run_xgboost_imdb
from utils import remove_string_cols
from utils import classify
from utils import fill_nan
from utils import clean_backward_title

def data_prepocessing():
    df = pd.read_csv('../data/movie_metadata.csv', error_bad_lines=False)
    #df = df.fillna(value=0,axis=1)

    cols = list(df.columns)
    df = fill_nan(df,cols)

    df['movie_title'] = df['movie_title'].astype(str)
    df['movie_title'] = df['movie_title'].apply(clean_backward_title)

    col = list(df.describe().columns)
    sc = StandardScaler()
    # sc = MinMaxScaler()
    temp = sc.fit_transform(df[col])
    df[col] = temp
    df_standard = df[list(df.describe().columns)]
    df_standard.columns
    return(df, df_standard)

if __name__ == '__main__':
    df, df_standard = data_prepocessing()
    run_pca(df_standard, df)

    # remove all irrelevant columns
    df = df.drop('facenumber_in_poster', 1)
    df = df.drop('title_year', 1)
    df = df.drop('aspect_ratio', 1)
    df = df.drop('duration', 1)

    df_knn = df

    df_knn["class"] = df_knn.apply(classify, axis=1)
    df_knn = df_knn.drop('imdb_score', 1)
    df_knn = remove_string_cols(df_knn)
    #cols = list(df_knn.columns)
    #df_knn = df_knn.fillna(value=df_knn[cols].median(),axis=1)
    df_knn = df_knn.fillna(value=0,axis=1)
    run_knn(df_knn)

    run_random_forest(df_knn)

    run_logistic_regression()

    run_xgboost_cornell()
    run_xgboost_imdb(df_knn)
