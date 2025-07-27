# Load EDA Pkgs
import pandas as pd
import neattext.functions as nfx

# Load ML/Rc Pkgs
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity,linear_kernel

from google.colab import files

u=files.upload()
df=pd.read_csv('udemy_courses.csv')
df.head(50)

df.head()

df['course_title']

dir(nfx)

# Clean Text:stopwords,special charac
df['clean_course_title'] = df['course_title'].apply(nfx.remove_stopwords)

# Clean Text:stopwords,special charac
df['clean_course_title'] = df['clean_course_title'].apply(nfx.remove_special_characters)

df[['course_title','clean_course_title']]

# Vectorize our Text
count_vect = CountVectorizer()
cv_mat = count_vect.fit_transform(df['clean_course_title'])

# Sparse
cv_mat

# Dense
cv_mat.todense()

df_cv_words = pd.DataFrame(cv_mat.todense(),columns=count_vect.get_feature_names_out())

df_cv_words.head()

# Cosine Similarity Matrix
cosine_sim_mat = cosine_similarity(cv_mat)

cosine_sim_mat

df.head()

# Get Course ID/Index
course_indices = pd.Series(df.index,index=df['course_title']).drop_duplicates()

course_indices

course_indices['How To Maximize Your Profits Trading Options']

idx = course_indices['How To Maximize Your Profits Trading Options']

idx

scores = list(enumerate(cosine_sim_mat[idx]))


scores

# Sort our scores per cosine score
sorted_scores = sorted(scores,key=lambda x:x[1],reverse=True)

# Omit the First Value/itself
sorted_scores[1:]


# Selected Courses Indices
selected_course_indices = [i[0] for i in sorted_scores[1:]]

selected_course_indices

# Selected Courses Scores
selected_course_scores = [i[1] for i in sorted_scores[1:]]

recommended_result = df['course_title'].iloc[selected_course_indices]

rec_df = pd.DataFrame(recommended_result)

rec_df.head()

rec_df['similarity_scores'] = selected_course_scores

rec_df


def recommend_course(title,num_of_rec=10):
    # ID for title
    idx = course_indices[title]
    # Course Indice
    # Search inside cosine_sim_mat
    scores = list(enumerate(cosine_sim_mat[idx]))
    # Scores
    # Sort Scores
    sorted_scores = sorted(scores,key=lambda x:x[1],reverse=True)
    # Recomm
    selected_course_indices = [i[0] for i in sorted_scores[1:]]
    selected_course_scores = [i[1] for i in sorted_scores[1:]]
    result = df['course_title'].iloc[selected_course_indices]
    rec_df = pd.DataFrame(result)
    rec_df['similarity_scores'] = selected_course_scores
    return rec_df.head(num_of_rec)




recommend_course('Options Trading - How to Win with Weekly Options',20)











