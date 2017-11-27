from flask import Flask, redirect, render_template, request, session, abort
from Movie_success_prediction import movie_success

app = Flask(__name__)
 
@app.route("/")
def hello():
	return render_template('test.html')

@app.route("/home", methods=['POST'])
def form():
	num_critic_for_reviews = float(request.form['num_critic_for_reviews'])
	duration = float(request.form['duration'])
	director_facebook_likes = float(request.form['director_facebook_likes'])
	actor_3_facebook_likes = float(request.form['actor_3_facebook_likes'])
	actor_1_facebook_likes = float(request.form['actor_1_facebook_likes'])
	num_voted_users = float(request.form['num_voted_users'])
	cast_total_facebook_likes = float(request.form['cast_total_facebook_likes'])
	facenumber_in_poster = float(request.form['facenumber_in_poster'])
	num_user_for_reviews = float(request.form['num_user_for_reviews'])
	budget = float(request.form['budget'])
	title_year = float(request.form['title_year'])
	actor_2_facebook_likes = float(request.form['actor_2_facebook_likes'])
	aspect_ratio = float(request.form['aspect_ratio'])
	movie_facebook_likes = float(request.form['movie_facebook_likes'])


	pred = movie_success(num_critic_for_reviews, duration, director_facebook_likes, actor_3_facebook_likes, 
					  actor_1_facebook_likes, num_voted_users, cast_total_facebook_likes, facenumber_in_poster, 
					  num_user_for_reviews, budget, title_year, actor_2_facebook_likes, aspect_ratio, movie_facebook_likes)
	
	return render_template('home.html', pred = pred)

if __name__ == "__main__":
	app.debug = True
	app.run()