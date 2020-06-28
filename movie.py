import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

@st.cache
def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]

df= pd.read_csv('movie_dataset.csv')
features = ['keywords','cast','genres','director']
for feature in features:
	df[feature] = df[feature].fillna('')

def combine_features(row):
	
		return row['keywords'] +" "+row['cast']+" "+row["genres"]+" "+row["director"]
		

df["combined_features"] = df.apply(combine_features,axis=1)


cv = CountVectorizer()

count_matrix = cv.fit_transform(df["combined_features"])
cosine_sim = cosine_similarity(count_matrix) 

st.title("Movie Recommendation site")
st.header('Please make sure to give the full and correct name of the movie')
st.text('The movies in the database are case sensitive.Please specify the exact name')

movie_user_likes = st.text_input('Search for your movie here','Titanic')
try :
    movie_index = get_index_from_title(movie_user_likes)
    similar_movies =  list(enumerate(cosine_sim[movie_index]))
    sorted_similar_movies = sorted(similar_movies,key=lambda t:t[1],reverse=True)
    x=[]
    i=0
    for movie in sorted_similar_movies:
        x.append((get_title_from_index(movie[0])))
        i=i+1
        if i>10 :
            break
	#Loop Ends Here		
    Moviedb = pd.DataFrame(x)
    Moviedb.columns=['Top 10 Movies']
    Moviedb.drop([0],axis=0,inplace=True)
    if st.button('Recommend'):
      st.balloons()
      st.write('Top 10 Recommended movies')
      st.dataframe(Moviedb)
except :
    st.write('Uh Oh It seems you might have mispelt the movie')




