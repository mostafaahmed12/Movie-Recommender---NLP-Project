from itertools import combinations
from skimage.metrics import mean_squared_error
from sklearn import naive_bayes, model_selection, metrics, linear_model
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import plotly.express as px
from matplotlib import pyplot as plt
import seaborn as sns
import re
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import squareform, pdist
from sklearn.preprocessing import StandardScaler, MinMaxScaler


movies = pd.read_csv('C:/Users/Dragonoid/Desktop/Movie recommender/movies.csv', sep=';', encoding='latin-1').drop('Unnamed: 3', axis=1)
users = pd.read_csv('C:/Users/Dragonoid/Desktop/Movie recommender/users.csv', sep=';', encoding='latin-1')
ratings = pd.read_csv('C:/Users/Dragonoid/Desktop/Movie recommender/ratings.csv', sep=';', encoding='latin-1')


# Creating merged data frame for using in multiple functions
df = pd.merge(movies, ratings, on='movieId')
df = pd.merge(df, users, on='userId')
df = df.dropna()

# Split the movie titles from their years
movies['year'] = movies['title'].str.extract(r'\((\d{4})\)$')  # Extract the year in parentheses
movies['title'] = movies['title'].str.replace(r'\s*\(\d{4}\)$', '', regex=True)  # Remove the year from the title
movies['title'] = movies['title'].str.replace(' ', '').str.lower()

# Pre-processing
accepted_genres = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama',
                   'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War',
                   'Western']

movies_filtered = movies[movies['genres'].apply(lambda x: set(x.split('|')).issubset(set(accepted_genres)))]
ratings_filtered = ratings[ratings['movieId'].isin(movies_filtered['movieId'])]

# Normalize the ratings column
ratings_normalized = ratings_filtered.copy()
ratings_normalized['rating'] = (ratings_filtered['rating']) * 2

# One hot encoding
one_hot = movies_filtered['genres'].str.get_dummies(sep='|')
movies_encoded = pd.concat([movies_filtered, one_hot], axis=1)

# Fixing the movie titles with regular expressions
def convert_title(title):
    pattern = r"(.+?),\s*(the)$"
    match = re.search(pattern, title, flags=re.IGNORECASE)
    if match:
        new_title = match.group(2).strip() + ' ' + match.group(1).strip()
        return new_title
    else:
        return title


title = movies_encoded['title'].astype(str)
movies_encoded['title'] = title.apply(convert_title)

# Making a matrix of the userId and movieID
final_dataset = ratings_normalized.pivot(index='movieId', columns='userId', values='rating')

# Converting the NaN values to 0
final_dataset.fillna(0, inplace=True)


# Applying the machine learning algorithm to the data
nn_algo = NearestNeighbors(metric='cosine')
nn_algo.fit(final_dataset)

############################## TRAINING AND TESTING ###############################

df = pd.merge(movies_encoded, ratings_filtered, on='movieId')
# df = pd.merge(df, users, on='userId')
df = df.dropna()
data = df.copy()
# print(df.head())


# creating the user matrix
df = df.drop(['title', 'timestamp', 'genres'], axis=1)
# Split the data into train and test sets
Y = df['rating']
X = df.drop('rating', axis=1)
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=42)


def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    classifier.fit(feature_vector_train, label)
    predictions = classifier.predict(feature_vector_valid)
    m = mean_squared_error(y_test, predictions)
    return metrics.accuracy_score(predictions, y_test), m


scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

scale = StandardScaler()
x_log_train = scale.fit_transform(x_train_scaled)
x_log_test = scale.transform(x_test_scaled)

accuracy, m = train_model(naive_bayes.MultinomialNB(alpha=0.5), x_train_scaled, y_train, x_test_scaled)
print("naive Accuracy: ", accuracy)
print("mean squared error of naive", m)

accuracy, m = train_model(linear_model.LogisticRegression(), x_log_train, y_train, x_log_test)
print("logistic Accuracy: ", accuracy)
print("mean squared error of logistic", m)

# Movie recommender function
def recommend_on_movie(movie, n_recommend=10):
    # Clean up the input movie title by removing any spaces
    movie_clean = movie.lower()

    # Find matches in the movies_filtered DataFrame
    matches = movies_filtered[movies_filtered['title'] == movie_clean]
    if len(matches) == 1:
        movieid = int(matches['movieId'].iloc[0])
    else:
        # Handle the case where there are zero or multiple matches
        return []

    final_dataset.fillna(0, inplace=True)
    distance, neighbors = nn_algo.kneighbors([final_dataset.loc[movieid]], n_neighbors=n_recommend + 1)
    movieids = final_dataset.iloc[neighbors[0]].index
    recommends = movies_filtered[movies_filtered['movieId'].isin(movieids)]['title'].tolist()
    return recommends





# Creating a unique list of genres from the movies_filtered DataFrame
movies_filtered_unique = movies_filtered.drop_duplicates('title')

# Reset the index of movies_filtered_unique
movies_filtered_unique = movies_filtered_unique.reset_index(drop=True)

# Creating a TF-IDF matrix
tf = TfidfVectorizer(analyzer=lambda s: (c for i in range(1, 4) for c in combinations(s.split('|'), r=i)))
tfidf_matrix = tf.fit_transform(movies_filtered_unique['genres'])

# Compute the pairwise cosine similarities
cosine_sim = pdist(tfidf_matrix.toarray(), metric='cosine')

# Create a square similarity matrix
similarity_matrix = pd.DataFrame(squareform(1 - cosine_sim), index=movies_filtered_unique['title'], columns=movies_filtered_unique['title'])

# Sample a subset of the similarity matrix
sample_similarity = similarity_matrix.sample(5, axis=1).sample(10, axis=0)

print(similarity_matrix)





def genre_recommendations(title, M, items, k=10):
    """
    Recommends movies based on a similarity dataframe
    Parameters
    ----------
    title : str
        Movie title (index of the similarity dataframe)
    M : pd.DataFrame
        Similarity dataframe, symmetric, with movies as indices and columns
    items : pd.DataFrame
        Contains both the title and some other features used to define similarity
    k : int
        Amount of recommendations to return
    """
    ix = M.loc[title].to_numpy().argpartition(range(-1, -k, -1))
    closest = M.columns[ix[-1:-(k+2):-1]]
    closest = closest.drop(title, errors='ignore')
    return pd.DataFrame(closest).merge(items).head(k)


# Test the movie recommender system on the user input

title = input("Enter your movie title please?")
title = title.lower().replace(" ", "")
type = input("enter 1 for user collaborative filtering or 2 for content-based recommendation: ")

if type == '1':
    recommendations = recommend_on_movie(title)
    if len(recommendations) == 0 :
        print(f"Sorry, could not find any recommendations for {title}.\n")
    else:
        print(f"Top 10 Recommended Movies: {recommendations}\n")
else:
    recommendations = genre_recommendations(title, similarity_matrix, movies_filtered_unique)
    if len(recommendations) == 0 :
        print(f"Sorry, could not find any recommendations for {title}.\n")
    else:
        print(f"Top 10 Recommended Movies: {recommendations}\n")


############################## visualization #####################################

# review of the most freqent rating
plt.subplot(1, 2, 1)
plt.title('Distribution of ratings')
data['rating'].value_counts().head().plot(kind='pie', autopct='%1.1f%%', figsize=(10, 8)).legend()

# review of the most popular movies
# plt.subplot(1, 2, 2)
movie_counts = data['title'].value_counts().head(15)

# Create a bar plot to visualize the most popular movies
plt.figure(figsize=(10, 6))
sns.barplot(x=movie_counts.index, y=movie_counts.values)
plt.title('Most Popular Movies')
plt.xlabel('Movie Title')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()

# plotting a distribution of the ratings and the genres
data['genres'] = data['genres'].apply(lambda x: x.split('|'))
df_exploded = data.explode('genres')
histogram = px.histogram(df_exploded, x='genres', height=400, title='Movie count by genre')
histogram.update_xaxes(categoryorder="total descending")
histogram.show()