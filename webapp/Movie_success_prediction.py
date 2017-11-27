import numpy as np
import re
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from requests import get
import unicodedata
from bs4 import BeautifulSoup
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import accuracy_score
import sys
reload(sys)
sys.setdefaultencoding('utf-8')



def movie_success(num_critic_for_reviews, duration, director_facebook_likes, actor_3_facebook_likes, 
									actor_1_facebook_likes, num_voted_users, cast_total_facebook_likes, facenumber_in_poster, 
									num_user_for_reviews, budget, title_year, actor_2_facebook_likes, aspect_ratio, movie_facebook_likes):

		df = pd.read_csv('movie_metadata.csv')


		def classify(col):
				if col['imdb_score'] >= 0 and col['imdb_score'] < 4:
						return 0
				elif col['imdb_score'] >= 4 and col['imdb_score'] < 6:
						return 1
				elif col['imdb_score'] >= 6 and col['imdb_score'] < 7:
						return 2
				elif col['imdb_score'] >= 7 and col['imdb_score'] < 8:
						return 3
				elif col['imdb_score'] >= 8 and col['imdb_score'] <= 10:
						return 4


		df['Success'] = df.apply(classify, axis=1)



		def fill_nan(col):
				df[col] = df[col].fillna(df[col].median())

		cols = list(df.columns)
		fill_nan(cols)



		def clean_backward_title(col):
				string = col.rstrip()[:-2]
				return unicodedata.normalize('NFD', unicode(string, 'utf-8')).encode('ascii', 'ignore')


		df['movie_title'] = df['movie_title'].astype(str)
		df['movie_title'] = df['movie_title'].apply(clean_backward_title)



		col = list(df.describe().columns)
		col.remove('Success')




		sc = StandardScaler()
		temp = sc.fit_transform(df[col])
		df[col] = temp




		features = col
		features.remove('imdb_score')
		features.remove('gross')



		# X_train, X_test, y_train, y_test = train_test_split(df[features], df['Success'], test_size=0.2)

		# rf = RandomForestClassifier(random_state=1, n_estimators=250, min_samples_split=8, min_samples_leaf=4)

		rf = GradientBoostingClassifier(random_state=0, n_estimators=250, min_samples_split=8, 
																	 min_samples_leaf=4, learning_rate=0.1)

		# rf = xgb.XGBClassifier(n_estimators=250)

		rf.fit(df[features], df['Success'])

		test_data = pd.DataFrame({'num_critic_for_reviews': num_critic_for_reviews, 
											'duration': duration, 
											'director_facebook_likes': director_facebook_likes, 
											'actor_3_facebook_likes': actor_3_facebook_likes, 
											'actor_1_facebook_likes': actor_1_facebook_likes, 
											'num_voted_users': num_voted_users, 
											'cast_total_facebook_likes': cast_total_facebook_likes, 
											'facenumber_in_poster': facenumber_in_poster, 
											'num_user_for_reviews': num_user_for_reviews, 
											'budget': budget, 
											'title_year': title_year, 
											'actor_2_facebook_likes': actor_2_facebook_likes, 
											'aspect_ratio': aspect_ratio, 
											'movie_facebook_likes': movie_facebook_likes}, index=[0])

		# temp_test_data = sc.fit_transform(test_data)

		# print temp_test_data

		predictions = rf.predict(pd.DataFrame(test_data))

		predictions = predictions.astype(int)


		return predictions
