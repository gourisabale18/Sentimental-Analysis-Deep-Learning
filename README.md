## Sentimental-Analysis-Deep-Learning

Through this project we are aiming to perform sentimental analysis of student's comments on RateMyProfessors website https://www.ratemyprofessors.com/. Our primary goal is to predict the quality and difficulty scores for professor's teaching using recurrent neural network(RNN) in deep learning.
This information would help future students to take the review the course and its professor's teaching before taking the course.

### Tech Stack Used
1. Python
2. Recurrent Neural Networks, LSTM, GRUs 
3. Keras, TensorFlow
4. Data science libraries such as Numpy,Pandas
5. Matplotlib
6. Scikitlearn
7. Jupyter Notebook
8. Google Collab
9. One hot encoder, Glove embeddings, word2vec

### Tasks we performed:
1. Data Collection
2. Data Preprocessing
3. Word Emebeddings using glove and word2vec
4. Split the datasets into training and testing sets
5. Train the main model
6. Analyse the results by comparing the models and select the best model for prediction

### Steps to execute the project: 

1. Download the ipynb file.
2. This file already contains the results of sentimental analysis done to predict the quality and difficulty of scores.
3. If you want to rerun the entire file, copy the code block by block in google colab notebook /jupyter notebook and run it.
4. Copy the json file and csv file in root folder to local drive of colab for accessing data.
5. Run the code block by block to see the result.


   
Sentimental Analysis using Recurrent Neural Networks
Main Objective
Through this project we are performing sentimental analysis of comments using sample datasets of RateMyProfessors website. Our primary goal is to predict the quality and difficulty scores for professor's teaching based on the comments provided by students.

Tasks we performed:
Data Collection
Data Preprocessing
Word Emebeddings using glove and word2vec
Split the datasets into training and testing sets
Train the main model
Analyse the results by comparing the models and select the best model for prediction
Task 1 : Data Collection
Mount new drive into google colab and add datasets into new drive

from google.colab import drive
drive.mount('/content/drive')
     
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
Read data from CSV File RateMyProfessor_Sample data.csv and store it into Panda Dataframe

csvFile = '/content/drive/MyDrive/RateMyProfessor_Sample data.csv'
import pandas as pd
     
Read columns for quality ,difficulty and comments and store it into Panda dataframe


df1 = pd.read_csv(csvFile, usecols=['comments','student_star', 'student_difficult'])
     
Print first 5 rows of dataframe

df1.head()
     
student_star	student_difficult	comments
0	5.0	3.0	This class is hard, but its a two-in-one gen-e...
1	5.0	2.0	Definitely going to choose Prof. Looney\'s cla...
2	4.0	3.0	I overall enjoyed this class because the assig...
3	5.0	3.0	Yes, it\'s possible to get an A but you\'ll de...
4	5.0	1.0	Professor Looney has great knowledge in Astron...
Rename Columns of dataframe to maintain consistency

df1 = df1.rename(columns={'comments': 'comments', 'student_star': 'quality', 'student_difficult': 'difficulty'})
     
Verify column change and data


df1.head()
     
quality	difficulty	comments
0	5.0	3.0	This class is hard, but its a two-in-one gen-e...
1	5.0	2.0	Definitely going to choose Prof. Looney\'s cla...
2	4.0	3.0	I overall enjoyed this class because the assig...
3	5.0	3.0	Yes, it\'s possible to get an A but you\'ll de...
4	5.0	1.0	Professor Looney has great knowledge in Astron...
Fetch the count of missing values


     

df1.isnull().sum()
     
quality       5
difficulty    5
comments      7
dtype: int64
Use glom library for data transformation

!pip install glom
from glom import glom
     
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Requirement already satisfied: glom in /usr/local/lib/python3.10/dist-packages (23.3.0)
Requirement already satisfied: boltons>=19.3.0 in /usr/local/lib/python3.10/dist-packages (from glom) (23.0.0)
Requirement already satisfied: face==20.1.1 in /usr/local/lib/python3.10/dist-packages (from glom) (20.1.1)
Requirement already satisfied: attrs in /usr/local/lib/python3.10/dist-packages (from glom) (23.1.0)
Read json file all_reviews.json and store it in panda dataframe This will be our second dataset


jsonFile = '/content/drive/MyDrive/all_reviews.json'
df2 = pd.read_json(jsonFile)

     

df2 = pd.DataFrame({
    'comments': df2[0].apply(lambda row: glom(row, 'Comment')), 
    'quality': df2[0].apply(lambda row: glom(row, 'Quality')), 
    'difficulty': df2[0].apply(lambda row: glom(row, 'Difficulty'))})

     
Display contents of second dataset

df2.head()
     
comments	quality	difficulty
0	Professor Nichols is super nice and very whole...	5.0	2.0
1	Prof Humes is really knowledgeable, but her le...	4.0	4.0
2	If you fall behind it's almost impossible to c...	3.0	4.0
3	Prof. Meyer is intensely brilliant and provoke...	4.5	4.0
4	Would not recommend taking him for fhs or even...	1.0	4.0

df2.info()
     
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 288 entries, 0 to 287
Data columns (total 3 columns):
 #   Column      Non-Null Count  Dtype 
---  ------      --------------  ----- 
 0   comments    288 non-null    object
 1   quality     288 non-null    object
 2   difficulty  288 non-null    object
dtypes: object(3)
memory usage: 6.9+ KB

df2['quality'] = df2['quality'].astype('float32')
df2['difficulty'] = df2['difficulty'].astype('float32')
     
Check for null values in dataframe

df2.isnull().sum()
     
comments      0
quality       0
difficulty    0
dtype: int64
Create a dataframe using web scrapping

!pip install RateMyProfessorAPI 
     
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Requirement already satisfied: RateMyProfessorAPI in /usr/local/lib/python3.10/dist-packages (1.3.4)
Requirement already satisfied: lxml in /usr/local/lib/python3.10/dist-packages (from RateMyProfessorAPI) (4.9.2)
Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from RateMyProfessorAPI) (2.27.1)
Requirement already satisfied: beautifulsoup4 in /usr/local/lib/python3.10/dist-packages (from RateMyProfessorAPI) (4.11.2)
Requirement already satisfied: soupsieve>1.2 in /usr/local/lib/python3.10/dist-packages (from beautifulsoup4->RateMyProfessorAPI) (2.4.1)
Requirement already satisfied: charset-normalizer~=2.0.0 in /usr/local/lib/python3.10/dist-packages (from requests->RateMyProfessorAPI) (2.0.12)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->RateMyProfessorAPI) (2022.12.7)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->RateMyProfessorAPI) (3.4)
Requirement already satisfied: urllib3<1.27,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->RateMyProfessorAPI) (1.26.15)

import ratemyprofessor
professor = ratemyprofessor.get_professor_by_school_and_name(ratemyprofessor.get_school_by_name('California State University-Fullerton'),'Kenytt Avery')
     
13)Display the contents of this dataset


print("Professor name = ",professor.name)
print("Professor Ratings =", professor.rating)
print("Professor Difficulty=", professor.difficulty)
     
Professor name =  Kenytt Avery
Professor Ratings = 3.6
Professor Difficulty= 4

comments = []
quality = []
difficulty = []
for rating in professor.get_ratings():
  comments.append(rating.comment)
  quality.append(rating.rating)
  difficulty.append(rating.difficulty)
print("First 5 comments")
print(comments[:5])
print("First 5 ratings")
print(quality[:5])
print("First 5 difficulty")
print(difficulty[:5])
     
First 5 comments
["He teaches NOTHING during lecture, no code, no helpful lecture. Not Grading for the whole semester and he'll grade it after final project. Member of project are random. Project 1 is at almost first half semester, and Project 2,3,4 are on the last month of semester WHYY. Please AVOID at all COST and don't believe in positive review.", "One of the best professors at csuf for cs department in my opinion. Just make sure to keep up with the learning material and don't miss any assignments because they build on top of each other concept wise. ", 'He does not grade until the end of the semester, until then you have no idea what your grade will be. Only group projects built around one big project, final group was randomized for no reason. He doesnt teach ANY code during lecture, only office hours (essential for projects). Lectures are useless, better off learning from youtube. AVOID PLEASE.\n', "Very knowledgeable about the topic he teaches, however the lectures are relatively dry. There are two weeks left till the semester ends and he still hasn't graded a single thing like what the other reviews said. All group projects are randomized, you get new members for every project, so good luck if you get someone who doesn't contribute anything.", "He doesn't grade anything all semester, so you have no idea if you're even doing projects correctly. He says the class is designed for beginners, but the very first project is literally back-end game development that just feels overwhelming when learning about APIs. Semester long projects with group member rotation, useless lectures."]
First 5 ratings
[1, 5, 1, 2, 2]
First 5 difficulty
[5, 4, 5, 4, 5]

dataframe = {'comments':comments,'quality':quality,'difficulty':difficulty}
df3 = pd.DataFrame(dataframe)
df3.head()
     
comments	quality	difficulty
0	He teaches NOTHING during lecture, no code, no...	1	5
1	One of the best professors at csuf for cs depa...	5	4
2	He does not grade until the end of the semeste...	1	5
3	Very knowledgeable about the topic he teaches,...	2	4
4	He doesn't grade anything all semester, so you...	2	5
Change the data type of quality and difficulty attributes to float32

df3['quality'] = df3['quality'].astype('float32')
df3['difficulty'] = df3['difficulty'].astype('float32')
     

df3.info()
     
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 50 entries, 0 to 49
Data columns (total 3 columns):
 #   Column      Non-Null Count  Dtype  
---  ------      --------------  -----  
 0   comments    50 non-null     object 
 1   quality     50 non-null     float32
 2   difficulty  50 non-null     float32
dtypes: float32(2), object(1)
memory usage: 928.0+ bytes
Combine all 3 panda dataframes to create one single dataframe

df = df1.append(df2)
df = df.append(df3)
df.head()
     
<ipython-input-301-e542b627fe56>:1: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
  df = df1.append(df2)
<ipython-input-301-e542b627fe56>:2: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
  df = df.append(df3)
quality	difficulty	comments
0	5.0	3.0	This class is hard, but its a two-in-one gen-e...
1	5.0	2.0	Definitely going to choose Prof. Looney\'s cla...
2	4.0	3.0	I overall enjoyed this class because the assig...
3	5.0	3.0	Yes, it\'s possible to get an A but you\'ll de...
4	5.0	1.0	Professor Looney has great knowledge in Astron...
Verify total no of rows in a dataframe

df.shape
     
(20338, 3)
Check for null values and drop them if exist

df.isnull().sum()
     
quality       5
difficulty    5
comments      7
dtype: int64

df.dropna(axis=0,inplace=True)
     

df.shape
     
(20331, 3)

df.reset_index(inplace=True)
     
Task 2: Data Preprocessing
Remove digits
Remove extra spaces
Remove emojis
Convert to lower case
Lemmalization
Store in list
Deal with missing data
Import all dependencies

#import packages for regular expressions and emojis
!pip install contractions
import re
!pip install emoji
import emoji
     
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Requirement already satisfied: contractions in /usr/local/lib/python3.10/dist-packages (0.1.73)
Requirement already satisfied: textsearch>=0.0.21 in /usr/local/lib/python3.10/dist-packages (from contractions) (0.0.24)
Requirement already satisfied: anyascii in /usr/local/lib/python3.10/dist-packages (from textsearch>=0.0.21->contractions) (0.3.2)
Requirement already satisfied: pyahocorasick in /usr/local/lib/python3.10/dist-packages (from textsearch>=0.0.21->contractions) (2.0.0)
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Requirement already satisfied: emoji in /usr/local/lib/python3.10/dist-packages (2.2.0)

import re 
from nltk.stem import WordNetLemmatizer
import gensim
import nltk
import contractions
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
     
[nltk_data] Downloading package wordnet to /root/nltk_data...
[nltk_data]   Package wordnet is already up-to-date!
[nltk_data] Downloading package omw-1.4 to /root/nltk_data...
[nltk_data]   Package omw-1.4 is already up-to-date!
[nltk_data] Downloading package punkt to /root/nltk_data...
[nltk_data]   Package punkt is already up-to-date!
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
Defining functions to clean Data (As a part of preprocessing)

def remove_links(text):
  return re.sub(r"(?:\@|https?\://)\S+", "", str(text))

def remove_email(text):
  return re.sub(r'\S+@\S+', '', text)

def remove_extra_spaces(text):
  return re.sub(' +',' ',text)

def remove_extra_line(text):
  return re.sub('\n+',' ',text)

def expand_contractions(text):
    """expand shortened words, e.g. don't to do not"""
    text = contractions.fix(text)
    return text

def remove_digits(text):
  return re.sub('[^a-zA-Z]',' ',text)

def convert_lower(text):
  return text.lower()

def tokenize(text):
  return text.split()

     
Define function to remove stopwords

#fetch stopwords from predefined library
stop_words = stopwords.words('english')

#write a function to remove stopwords
def remove_stop_words(data): 
  filtered_words=[]
  for word in data:
    if word not in stop_words:
      filtered_words.append(word)

  return filtered_words
     
Apply preprocessing techniques and perform lemmatization by removing stopwords

#Lemmelization
corpus=[]
for i in range(len(df)):
  text = remove_links(str(df['comments'][i]))
  text = remove_email(text)
  text = remove_extra_spaces(text)
  text = remove_extra_line(text)
  text = expand_contractions(text)
  text = remove_digits(text)
  text = convert_lower(text)
  text = tokenize(text)
  lm=WordNetLemmatizer()
  text = [lm.lemmatize(word) for word in text if word not in stopwords.words('english')]
  text = " ".join(text)
  corpus.append(text)
corpus[:5]
     
['class hard two one gen ed knockout content stimulating unlike class actually participate pas section easy offer extra credit every week funny dude much say',
 'definitely going choose prof looney class interesting class easy bring note exam need remember lot lot bonus point available observatory session awesome',
 'overall enjoyed class assignment straightforward interesting enjoy video project felt like one group cared enough help',
 'yes possible get definitely work content pretty interesting tog get super organized class multiple thing due every week ton lecture go possible would avoid class week course definitely always somethingto class',
 'professor looney great knowledge astronomy explain super easy way elementary class taught class great passion great illustration class definitely fun take interested knowledge class cover hesitate ask great teacher']
Task 3: Word Embedding
Convert words to vector
1a. Using GloVe
1b. Using word2vec
Train network
1a. Using Glove
Import the dependencies

import os
import urllib.request
import numpy as np
     

import tensorflow as tf
from tensorflow import keras
from tensorflow import keras
from tensorflow.keras.preprocessing import text
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tqdm import tqdm
     
Download pre-trained model for glove from https://nlp.stanford.edu/projects/glove/

urllib.request.urlretrieve('https://nlp.stanford.edu/data/glove.6B.zip','glove.6B.zip')
     
('glove.6B.zip', <http.client.HTTPMessage at 0x7f88b691c250>)

!unzip "/content/glove.6B.zip" -d "/content/"
     
Archive:  /content/glove.6B.zip
  inflating: /content/glove.6B.50d.txt  
  inflating: /content/glove.6B.100d.txt  
  inflating: /content/glove.6B.200d.txt  
  inflating: /content/glove.6B.300d.txt  
Create a dictionary containing every word and its corresponding vector.
Loop through the file and extract word and vectors

embedded_dictionary = {}
#with open('/content/glove.6B.200d.txt','r') as f:
with open('/content/glove.6B.200d.txt','r') as f:

  for line in f:
    newvalues = line.split()
    word = newvalues[0]
    vector = np.asarray(newvalues[1:],'float32')
    embedded_dictionary[word]=vector
     
Display embeddings for word bad

embedded_dictionary['bad']
     
array([ 1.9733e-01, -3.6513e-01, -4.4720e-02, -1.5762e-01, -5.2375e-01,
        1.2399e-01, -6.8523e-01, -3.3183e-01,  4.5466e-01,  6.6316e-01,
       -1.8251e-02,  2.1219e-01,  2.8740e-02, -1.4164e-01,  4.6345e-01,
        3.1726e-01, -5.8831e-01,  4.2796e-01,  4.5681e-02, -4.9805e-01,
        1.7164e-01,  2.4047e+00, -3.7963e-01,  8.2891e-02, -2.7570e-01,
       -6.8935e-01,  2.8693e-01,  1.3462e-01, -3.8978e-01, -3.6450e-01,
        1.6104e-01, -3.8947e-01, -6.7416e-01,  3.4876e-01, -2.5767e-01,
        7.6081e-02, -7.5393e-01, -4.2013e-01, -2.2229e-01,  8.3082e-02,
        4.5537e-01, -2.4032e-01, -2.4604e-01,  7.1795e-01,  4.3490e-01,
       -1.4638e-01,  2.5804e-01, -5.0251e-02, -2.0919e-01,  5.3911e-01,
        5.5961e-02,  8.5284e-02,  1.2690e-01, -1.1956e-01,  1.9526e-01,
       -1.0273e-01, -2.6263e-01,  1.5539e-01,  1.2326e-01, -3.6314e-01,
        1.1937e-01, -2.9376e-01, -4.3331e-01,  1.7554e-01, -1.9433e-01,
        4.2875e-01, -3.1927e-02,  7.6411e-02,  9.1824e-01,  5.2097e-01,
        4.5882e-01, -1.5501e-01, -3.3219e-01, -6.5191e-02,  1.6775e-01,
        2.1922e-01, -4.7281e-01,  2.3984e-01, -4.0465e-01, -1.0439e-01,
       -3.4691e-01,  1.5753e-01, -5.7694e-01, -4.7437e-02, -2.0240e-01,
       -6.5287e-01,  8.4951e-02, -1.3980e-01,  4.4506e-01, -9.8799e-01,
       -4.0171e-02, -1.4593e-01,  3.8093e-01,  6.1772e-01,  2.3097e-03,
        2.1583e-01, -7.4962e-01, -6.8691e-01,  4.6269e-02, -8.1172e-02,
       -3.8705e-03,  4.8277e-01,  1.7737e-02,  3.2582e-01, -7.4123e-02,
        5.6490e-01, -1.4126e-02,  9.4260e-01, -2.9498e-01,  2.7660e-01,
        4.4080e-01,  5.4514e-01,  1.4435e-01,  2.8680e-01,  2.1803e-01,
        2.4544e-01,  7.2372e-02,  6.5492e-02, -1.5449e-01, -3.3589e-01,
        7.7813e-01,  5.3097e-02,  1.7474e-01, -5.9337e-01, -2.9772e-01,
        1.4465e-01,  1.5864e-01,  3.9029e-01,  6.4841e-01, -4.4852e-02,
       -5.3986e-01, -5.3329e-02,  1.4798e-03,  1.2680e-01, -2.3332e-01,
       -1.0650e-01,  1.9711e-01,  3.7006e-01, -2.3177e-01, -2.3627e-01,
       -3.4470e-02,  1.0455e-01, -1.3961e-01, -4.4577e-01,  1.3341e+00,
       -5.6387e-02,  6.6159e-01, -9.2424e-01,  3.0929e-01,  5.7102e-01,
        1.1759e-01,  7.0285e-02, -9.4016e-02, -1.0825e-01, -1.6214e-01,
       -1.5485e-01, -1.5334e-01, -5.0790e-02, -6.9641e-02, -1.1453e-01,
        5.5958e-01,  2.1826e-01,  6.7110e-02, -1.3074e-01, -2.7369e-01,
       -1.5982e-01,  2.6641e-01,  8.5600e-01, -1.1592e-01,  1.4938e-02,
       -3.8181e-01,  2.4406e-01,  1.6237e-01,  1.0154e-01,  4.3720e-01,
        1.8502e-01, -4.5786e-01,  7.0190e-01,  2.9019e-01,  1.5988e-01,
        1.0185e+00, -4.9470e-02, -3.1153e-01,  4.5822e-01, -5.7704e-01,
       -2.6003e-01,  6.3846e-01, -4.7012e-03,  1.0199e-01,  2.2876e-01,
        1.7656e-01, -6.4874e-01, -3.4069e-01,  1.2860e-01,  4.4697e-01,
       -3.3129e-01, -4.6873e-02,  2.4106e-02, -1.6920e-01,  4.4173e-01],
      dtype=float32)
Find Maximum No of words from sequences of comments. We will use it as a parameter input length/sequence length while training the model

# split each sentence in the 'df['comments']' column into a list of words and find the maximum no of words from sentences
df['comments']=corpus
max_words = df['comments'].str.split().apply(len).max()

# print the maximum number of words
print('Maximum number of words:', max_words)
     
Maximum number of words: 76
Tokenize and Pad Sequences so that every sequence will have equal no of words

tokenizer_obj=Tokenizer()
tokenizer_obj.fit_on_texts(corpus)
sequences=tokenizer_obj.texts_to_sequences(corpus)

comments_pad=pad_sequences(sequences,maxlen=max_words,truncating='post',padding='post')
word_index=tokenizer_obj.word_index
print('Number of unique words:',len(word_index))


     
Number of unique words: 14403
Create embedding matrix using glove where every word will be represented with 200 dims

emedding_dim=200
num_words=len(word_index)+1
embedding_matrix=np.zeros((num_words,emedding_dim))
     

print(type(comments_pad))
print(comments_pad.dtype)

     
<class 'numpy.ndarray'>
int32
Converting feature vector of all inputs into dtype float32

new_comments = np.array(comments_pad, dtype=np.float32)
new_quality=np.array(df['quality'], dtype=np.float32)
new_difficulty=np.array(df['difficulty'],dtype=np.float32)

     
Verify the data types

print(new_comments.dtype)
print(new_quality.dtype)
print(new_difficulty.dtype)
     
float32
float32
float32
Check the shape of matrix

embedding_matrix.shape
     
(14404, 200)
Convert dataset into sequence of words and apply word embeddings on those sequences

#function to get word embedding
def get_embedding(word):
    if word in embedded_dictionary:
        return embedded_dictionary[word]
    else:
        return np.zeros((100,))
     
Task 4: Split Data into training and testing set : Using Glove Embeddings

Y = np.stack((new_quality, new_difficulty),axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(new_comments, Y, test_size=0.2)

     
Task 5: Train the Model using RNN and glove embeddings

from keras.models import Sequential
from keras.layers import Flatten,Input
from keras.layers import LSTM,Dense,Dropout,Embedding,CuDNNLSTM,Bidirectional,SpatialDropout1D,BatchNormalization,GRU
from keras.initializers import Constant
from keras.regularizers import l2
     

model1 = keras.Sequential([
    keras.layers.Embedding(num_words, 200, embeddings_initializer=Constant(embedding_matrix),
                   input_length=max_words),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.GRU(128, activation='relu', return_sequences=True),
    keras.layers.Dropout(0.3),
    keras.layers.GRU(64, activation='relu', return_sequences=True),
    keras.layers.Dropout(0.3),
    keras.layers.GRU(32, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1)
])

     
WARNING:tensorflow:Layer gru_30 will not use cuDNN kernels since it doesn't meet the criteria. It will use a generic GPU kernel as fallback when running on GPU.
WARNING:tensorflow:Layer gru_31 will not use cuDNN kernels since it doesn't meet the criteria. It will use a generic GPU kernel as fallback when running on GPU.
WARNING:tensorflow:Layer gru_32 will not use cuDNN kernels since it doesn't meet the criteria. It will use a generic GPU kernel as fallback when running on GPU.

model1.compile(optimizer='adam',loss="mse",metrics=['mse', 'mae'])

     

#Fit the model 
model1.fit(X_train, Y_train,batch_size=128,epochs=10,validation_split=0.1,verbose=1)
     
Epoch 1/10
115/115 [==============================] - 117s 959ms/step - loss: 12.6087 - mse: 12.6087 - mae: 3.2538 - val_loss: 12.1325 - val_mse: 12.1325 - val_mae: 3.1826
Epoch 2/10
115/115 [==============================] - 105s 918ms/step - loss: 11.8838 - mse: 11.8838 - mae: 3.1404 - val_loss: 11.4299 - val_mse: 11.4299 - val_mae: 3.0702
Epoch 3/10
115/115 [==============================] - 98s 852ms/step - loss: 11.1965 - mse: 11.1965 - mae: 3.0291 - val_loss: 10.7644 - val_mse: 10.7644 - val_mae: 2.9599
Epoch 4/10
115/115 [==============================] - 103s 894ms/step - loss: 10.5452 - mse: 10.5452 - mae: 2.9195 - val_loss: 10.1335 - val_mse: 10.1335 - val_mae: 2.8513
Epoch 5/10
115/115 [==============================] - 104s 904ms/step - loss: 9.9285 - mse: 9.9285 - mae: 2.8119 - val_loss: 9.5368 - val_mse: 9.5368 - val_mae: 2.7447
Epoch 6/10
115/115 [==============================] - 86s 749ms/step - loss: 9.3448 - mse: 9.3448 - mae: 2.7062 - val_loss: 8.9720 - val_mse: 8.9720 - val_mae: 2.6398
Epoch 7/10
115/115 [==============================] - 76s 663ms/step - loss: 8.7931 - mse: 8.7931 - mae: 2.6022 - val_loss: 8.4388 - val_mse: 8.4388 - val_mae: 2.5368
Epoch 8/10
115/115 [==============================] - 73s 634ms/step - loss: 8.2721 - mse: 8.2721 - mae: 2.5002 - val_loss: 7.9360 - val_mse: 7.9360 - val_mae: 2.4357
Epoch 9/10
115/115 [==============================] - 75s 647ms/step - loss: 7.7804 - mse: 7.7804 - mae: 2.3999 - val_loss: 7.4614 - val_mse: 7.4614 - val_mae: 2.3362
Epoch 10/10
115/115 [==============================] - 75s 645ms/step - loss: 7.3168 - mse: 7.3168 - mae: 2.3063 - val_loss: 7.0140 - val_mse: 7.0140 - val_mae: 2.2556
<keras.callbacks.History at 0x7f878be84340>

y_prediction= model1.predict(X_test)
     
382/382 [==============================] - 4s 6ms/step

y_prediction[:5]
     
array([[3.5273361],
       [3.5273361],
       [3.5273361],
       [3.5273361],
       [3.5273361]], dtype=float32)
Trained Model with Glove Embeddings that shows good results
By training this model with Bidirectional GRU's and by adding some dropout layers , we were able to achieve upto mse upto 0.4 and validation mse upto 0.8. This model is working good for glove emebddings.



from keras.layers import LSTM
model2= Sequential([
Embedding(num_words,200,input_length=max_words),
    SpatialDropout1D(0.4),
    Bidirectional(GRU(128,return_sequences=True)),
    Dropout(0.2),
    Bidirectional(GRU(64)),
    Dropout(0.2),
    Dense(64, activation="relu"),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(32,activation="relu"),
    Dense(2,activation="linear")
])
model2.summary()
     
Model: "sequential_9"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding_9 (Embedding)     (None, 76, 200)           3359800   
                                                                 
 spatial_dropout1d_3 (Spatia  (None, 76, 200)          0         
 lDropout1D)                                                     
                                                                 
 bidirectional_10 (Bidirecti  (None, 76, 256)          253440    
 onal)                                                           
                                                                 
 dropout_29 (Dropout)        (None, 76, 256)           0         
                                                                 
 bidirectional_11 (Bidirecti  (None, 128)              123648    
 onal)                                                           
                                                                 
 dropout_30 (Dropout)        (None, 128)               0         
                                                                 
 dense_27 (Dense)            (None, 64)                8256      
                                                                 
 dense_28 (Dense)            (None, 64)                4160      
                                                                 
 dropout_31 (Dropout)        (None, 64)                0         
                                                                 
 dense_29 (Dense)            (None, 32)                2080      
                                                                 
 dense_30 (Dense)            (None, 2)                 66        
                                                                 
=================================================================
Total params: 3,751,450
Trainable params: 3,751,450
Non-trainable params: 0
_________________________________________________________________

model2.compile(optimizer='adam',loss="mse",metrics=['mse', 'mae'])

     

model2.fit(X_train, Y_train,batch_size=128,epochs=10,validation_split=0.1,verbose=1)
     
Epoch 1/10
344/344 [==============================] - 49s 109ms/step - loss: 1.8759 - mse: 1.8759 - mae: 1.0835 - val_loss: 1.0765 - val_mse: 1.0765 - val_mae: 0.8218
Epoch 2/10
344/344 [==============================] - 15s 45ms/step - loss: 0.9888 - mse: 0.9888 - mae: 0.7822 - val_loss: 1.0394 - val_mse: 1.0394 - val_mae: 0.8216
Epoch 3/10
344/344 [==============================] - 13s 37ms/step - loss: 0.7892 - mse: 0.7892 - mae: 0.6882 - val_loss: 0.9187 - val_mse: 0.9187 - val_mae: 0.7624
Epoch 4/10
344/344 [==============================] - 11s 31ms/step - loss: 0.6634 - mse: 0.6634 - mae: 0.6223 - val_loss: 1.0040 - val_mse: 1.0040 - val_mae: 0.8087
Epoch 5/10
344/344 [==============================] - 10s 29ms/step - loss: 0.5917 - mse: 0.5917 - mae: 0.5803 - val_loss: 0.9058 - val_mse: 0.9058 - val_mae: 0.7570
Epoch 6/10
344/344 [==============================] - 10s 28ms/step - loss: 0.5337 - mse: 0.5337 - mae: 0.5458 - val_loss: 0.9039 - val_mse: 0.9039 - val_mae: 0.7564
Epoch 7/10
344/344 [==============================] - 10s 30ms/step - loss: 0.4851 - mse: 0.4851 - mae: 0.5153 - val_loss: 0.9713 - val_mse: 0.9713 - val_mae: 0.8000
Epoch 8/10
344/344 [==============================] - 10s 29ms/step - loss: 0.4529 - mse: 0.4529 - mae: 0.4922 - val_loss: 0.8726 - val_mse: 0.8726 - val_mae: 0.7446
Epoch 9/10
344/344 [==============================] - 9s 27ms/step - loss: 0.4208 - mse: 0.4208 - mae: 0.4693 - val_loss: 0.8746 - val_mse: 0.8746 - val_mae: 0.7545
Epoch 10/10
344/344 [==============================] - 10s 30ms/step - loss: 0.4023 - mse: 0.4023 - mae: 0.4563 - val_loss: 0.8187 - val_mse: 0.8187 - val_mae: 0.7208
<keras.callbacks.History at 0x7f88c117ec50>

#Predicted Labels
pred2 = model2.predict(X_test)
pred2[:5]
     
382/382 [==============================] - 5s 9ms/step
array([[3.2843676 , 3.0867293 ],
       [4.2647133 , 0.94531024],
       [3.6990047 , 3.513675  ],
       [1.3723811 , 2.6331584 ],
       [3.7967346 , 3.489458  ]], dtype=float32)

#Actual Labels
Y_test[:5]
     
array([[4., 4.],
       [5., 1.],
       [5., 3.],
       [2., 3.],
       [5., 5.]], dtype=float32)
Here predicted labels and actual labels are somewhat matching.

One hot encoding
One hot encoding is used to convert the sentences into the numerical values. Each word in each sentences converted ranging from 0 to total unique words in dataset.
After that, we performed the the padding to make each sentence of same length by finding maximum length of sentence

#importing preprocessing libraries
from keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
     

#performed one hot encoding
voc_size = len(set(corpus))
one_hot_repr = [one_hot(words,voc_size) for words in corpus]
one_hot_repr[:5]
     
[[6471,
  46292,
  45537,
  24437,
  22678,
  49604,
  17103,
  5966,
  5240,
  9340,
  6471,
  32791,
  7320,
  28489,
  37765,
  49357,
  9126,
  11149,
  3958,
  33541,
  35539,
  3255,
  47132,
  26023,
  31748],
 [46099,
  18533,
  8215,
  31414,
  5779,
  6471,
  44418,
  6471,
  49357,
  7780,
  13487,
  44949,
  24910,
  24945,
  25768,
  25768,
  2703,
  8799,
  32644,
  25470,
  49398,
  15845],
 [2977,
  8989,
  6471,
  22519,
  20114,
  44418,
  15215,
  27989,
  24568,
  4700,
  18358,
  24437,
  21889,
  12289,
  30665,
  3309],
 [14082,
  25816,
  8961,
  46099,
  29830,
  5966,
  44487,
  44418,
  493,
  8961,
  40353,
  10187,
  6471,
  1379,
  12573,
  23328,
  33541,
  35539,
  48813,
  2796,
  18569,
  25816,
  36008,
  1636,
  6471,
  35539,
  18651,
  46099,
  18791,
  46106,
  6471],
 [31248,
  5779,
  39874,
  41226,
  6124,
  4457,
  40353,
  49357,
  32036,
  12402,
  6471,
  19727,
  6471,
  39874,
  29816,
  39874,
  18855,
  6471,
  46099,
  26224,
  35322,
  40591,
  41226,
  6471,
  23209,
  19033,
  11679,
  39874,
  36121]]

#Performed padding
sent_len = max([len(sent) for sent in corpus])
embedded_docs = pad_sequences(one_hot_repr,padding="post",maxlen=sent_len)
embedded_docs[0]
     
array([ 6471, 46292, 45537, 24437, 22678, 49604, 17103,  5966,  5240,
        9340,  6471, 32791,  7320, 28489, 37765, 49357,  9126, 11149,
        3958, 33541, 35539,  3255, 47132, 26023, 31748,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0],
      dtype=int32)

#storing quality and difficulty in different list
rating = []
difficulty = []
for index,row in df.iterrows():
    if type(row["comments"]) == str:
        rating.append(row["quality"])
        difficulty.append(row["difficulty"])
print(len(rating))
print(len(difficulty))
     
60993
60993

#Preparing X and Y
import numpy as np
X = np.array(embedded_docs)
Y = np.stack((rating,difficulty),axis=1)
     

#Spliting the data into train and test by spliting ratio of 75-25%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state=42)

     

#import keras libraries
from keras.models import Sequential
from keras.layers import Flatten,Input
from keras.layers import LSTM,Dense,Dropout,Embedding,CuDNNLSTM,Bidirectional,SpatialDropout1D,BatchNormalization,GRU
from keras.initializers import Constant
from keras.regularizers import l2
     

#Build the model
model = Sequential([
    Embedding(voc_size,100,input_length=sent_len),
    Bidirectional(GRU(64,return_sequences=True)),
    Dropout(0.4),
    Bidirectional(GRU(32)),
    Dropout(0.4),
    Dense(128,activation="relu"),
    Dense(64,activation="relu"),
    Dropout(0.5),
    Dense(32,activation="relu"),
    Dense(2,activation="linear")
])
model.summary()
     
Model: "sequential_10"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding_10 (Embedding)    (None, 405, 100)          5014700   
                                                                 
 bidirectional_12 (Bidirecti  (None, 405, 128)         63744     
 onal)                                                           
                                                                 
 dropout_32 (Dropout)        (None, 405, 128)          0         
                                                                 
 bidirectional_13 (Bidirecti  (None, 64)               31104     
 onal)                                                           
                                                                 
 dropout_33 (Dropout)        (None, 64)                0         
                                                                 
 dense_31 (Dense)            (None, 128)               8320      
                                                                 
 dense_32 (Dense)            (None, 64)                8256      
                                                                 
 dropout_34 (Dropout)        (None, 64)                0         
                                                                 
 dense_33 (Dense)            (None, 32)                2080      
                                                                 
 dense_34 (Dense)            (None, 2)                 66        
                                                                 
=================================================================
Total params: 5,128,270
Trainable params: 5,128,270
Non-trainable params: 0
_________________________________________________________________

#compiled the model
model.compile(optimizer="adam",loss="mse")
     

#Fit the model on training data
history=model.fit(x_train,y_train,epochs=10,validation_split=0.1,verbose=1)
     
Epoch 1/10
1287/1287 [==============================] - 117s 83ms/step - loss: 1.7431 - val_loss: 1.3868
Epoch 2/10
1287/1287 [==============================] - 73s 57ms/step - loss: 0.9720 - val_loss: 1.1676
Epoch 3/10
1287/1287 [==============================] - 72s 56ms/step - loss: 0.7540 - val_loss: 1.1457
Epoch 4/10
1287/1287 [==============================] - 72s 56ms/step - loss: 0.6396 - val_loss: 0.9817
Epoch 5/10
1287/1287 [==============================] - 69s 54ms/step - loss: 0.5662 - val_loss: 0.9436
Epoch 6/10
1287/1287 [==============================] - 72s 56ms/step - loss: 0.5100 - val_loss: 0.9295
Epoch 7/10
1287/1287 [==============================] - 69s 54ms/step - loss: 0.4674 - val_loss: 0.8897
Epoch 8/10
1287/1287 [==============================] - 71s 55ms/step - loss: 0.4322 - val_loss: 0.8619
Epoch 9/10
1287/1287 [==============================] - 69s 54ms/step - loss: 0.4035 - val_loss: 0.7799
Epoch 10/10
1287/1287 [==============================] - 70s 55ms/step - loss: 0.3770 - val_loss: 0.8363

import matplotlib.pyplot as plt
# Plot the training and validation loss over each epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
     


#Make predictions
pred = model.predict(x_test)
pred[:5]
     
477/477 [==============================] - 14s 27ms/step
array([[1.8876605, 3.7087598],
       [3.9427278, 3.2785344],
       [3.3221188, 2.9812076],
       [2.4894245, 2.9957952],
       [3.56857  , 4.0621433]], dtype=float32)

#Actual values for comparison
print("Actual rating")
print(y_train[:5])

     
Actual rating
[[4.5 3. ]
 [2.5 4. ]
 [1.  3. ]
 [5.  3. ]
 [1.  5. ]]
As per the results the Mean square error for training is low 0.3770 and mean square error for validation is 0.8363 which is higher than the previous one. Also, the training loss is reducing exponentially while validation loss going reduced and went upward that shows overfitting

Data Augmentation
As you see the above results in overfitting, to avoid that we added the text augmentsation part using two methods. For that we have used inbuilt library nlpaug

Inserting Synonymes:
We inserted the synonames for each sentence in the dataframe by using SynonymAug function which inserts synonymes to maximum 3 words for each sentence

Swapping the random words in sentences:
We swapped the words from each sentence using RandomWordAug with action parameter as swap


!pip install nlpaug
     
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Requirement already satisfied: nlpaug in /usr/local/lib/python3.10/dist-packages (1.1.11)
Requirement already satisfied: gdown>=4.0.0 in /usr/local/lib/python3.10/dist-packages (from nlpaug) (4.6.6)
Requirement already satisfied: requests>=2.22.0 in /usr/local/lib/python3.10/dist-packages (from nlpaug) (2.27.1)
Requirement already satisfied: numpy>=1.16.2 in /usr/local/lib/python3.10/dist-packages (from nlpaug) (1.22.4)
Requirement already satisfied: pandas>=1.2.0 in /usr/local/lib/python3.10/dist-packages (from nlpaug) (1.5.3)
Requirement already satisfied: six in /usr/local/lib/python3.10/dist-packages (from gdown>=4.0.0->nlpaug) (1.16.0)
Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from gdown>=4.0.0->nlpaug) (4.65.0)
Requirement already satisfied: beautifulsoup4 in /usr/local/lib/python3.10/dist-packages (from gdown>=4.0.0->nlpaug) (4.11.2)
Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from gdown>=4.0.0->nlpaug) (3.12.0)
Requirement already satisfied: python-dateutil>=2.8.1 in /usr/local/lib/python3.10/dist-packages (from pandas>=1.2.0->nlpaug) (2.8.2)
Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas>=1.2.0->nlpaug) (2022.7.1)
Requirement already satisfied: charset-normalizer~=2.0.0 in /usr/local/lib/python3.10/dist-packages (from requests>=2.22.0->nlpaug) (2.0.12)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests>=2.22.0->nlpaug) (3.4)
Requirement already satisfied: urllib3<1.27,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests>=2.22.0->nlpaug) (1.26.15)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests>=2.22.0->nlpaug) (2022.12.7)
Requirement already satisfied: soupsieve>1.2 in /usr/local/lib/python3.10/dist-packages (from beautifulsoup4->gdown>=4.0.0->nlpaug) (2.4.1)
Requirement already satisfied: PySocks!=1.5.7,>=1.5.6 in /usr/local/lib/python3.10/dist-packages (from requests>=2.22.0->nlpaug) (1.7.1)

#import nlpaug library
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
     

#define the augmentors
aug = naw.SynonymAug(aug_src="wordnet",aug_max=3)
aug1 = naw.RandomWordAug(action='swap')
     

#defined two dataframes for augmentation
synaug_df = pd.DataFrame(columns=['comments','quality','difficulty'])
swapaug_df = pd.DataFrame(columns=['comments','quality','difficulty'])
     

#Augment by inserting synonymes
aug_text = []
for idx,row in df.iterrows():
  aug_data = {}
  aug_sent = aug.augment(row["comments"])
  aug_data["comments"] = aug_sent
  aug_data["quality"] = row["quality"]
  aug_data["difficulty"] = row["difficulty"]
  aug_data_df = pd.DataFrame([aug_data])
  synaug_df = pd.concat([synaug_df, aug_data_df], ignore_index = True)

     

synaug_df.shape
     
(20331, 3)

#Augment by swapping the words
aug_text = []
for idx,row in df.iterrows():
  aug_data = {}
  aug_sent = aug1.augment(row["comments"])
  aug_data["comments"] = aug_sent
  aug_data["quality"] = row["quality"]
  aug_data["difficulty"] = row["difficulty"]
  aug_data_df = pd.DataFrame([aug_data])
  swapaug_df = pd.concat([swapaug_df, aug_data_df], ignore_index = True)

     

swapaug_df.shape
     
(20331, 3)

#concate the dataframe 
df = pd.concat([df, synaug_df], ignore_index = True)
df = pd.concat([df, swapaug_df], ignore_index = True)
df.shape
     
(60993, 4)

df.head()
#df = df.drop(['index'],axis=1)
     
index	quality	difficulty	comments
0	0.0	5.0	3.0	This class is hard, but its a two-in-one gen-e...
1	1.0	5.0	2.0	Definitely going to choose Prof. Looney\'s cla...
2	2.0	4.0	3.0	I overall enjoyed this class because the assig...
3	3.0	5.0	3.0	Yes, it\'s possible to get an A but you\'ll de...
4	4.0	5.0	1.0	Professor Looney has great knowledge in Astron...

df.head()
     
index	quality	difficulty	comments
0	0.0	5.0	3.0	This class is hard, but its a two-in-one gen-e...
1	1.0	5.0	2.0	Definitely going to choose Prof. Looney\'s cla...
2	2.0	4.0	3.0	I overall enjoyed this class because the assig...
3	3.0	5.0	3.0	Yes, it\'s possible to get an A but you\'ll de...
4	4.0	5.0	1.0	Professor Looney has great knowledge in Astron...

df.isnull().sum()
     
index         40662
quality           0
difficulty        0
comments          0
dtype: int64

#performed data preprocessing
corpus=[]
for i in range(len(df)):
  text = remove_links(str(df['comments'][i]))
  text = remove_email(text)
  text = remove_extra_spaces(text)
  text = remove_extra_line(text)
  text = expand_contractions(text)
  text = remove_digits(text)
  text = convert_lower(text)
  text = tokenize(text)
  text = [lm.lemmatize(word) for word in text if word not in stopwords.words('english')]
  text = " ".join(text)
  corpus.append(text)
corpus[:5]
     
['class hard two one gen ed knockout content stimulating unlike class actually participate pas section easy offer extra credit every week funny dude much say',
 'definitely going choose prof looney class interesting class easy bring note exam need remember lot lot bonus point available observatory session awesome',
 'overall enjoyed class assignment straightforward interesting enjoy video project felt like one group cared enough help',
 'yes possible get definitely work content pretty interesting tog get super organized class multiple thing due every week ton lecture go possible would avoid class week course definitely always somethingto class',
 'professor looney great knowledge astronomy explain super easy way elementary class taught class great passion great illustration class definitely fun take interested knowledge class cover hesitate ask great teacher']

#Performed one hot encoding
voc_size = len(set(corpus))
one_hot_repr = [one_hot(words,voc_size) for words in corpus]
one_hot_repr[:5]
     
[[29452,
  12689,
  3570,
  37948,
  46982,
  27976,
  6037,
  3566,
  21567,
  10695,
  29452,
  39105,
  31282,
  20722,
  15501,
  25448,
  12639,
  8530,
  43198,
  11750,
  8495,
  30389,
  35672,
  12741,
  31820],
 [26577,
  48050,
  23621,
  16592,
  8134,
  29452,
  33814,
  29452,
  25448,
  21189,
  1652,
  8657,
  47960,
  34816,
  19184,
  19184,
  23306,
  46611,
  19347,
  29018,
  27079,
  8195],
 [45421,
  31188,
  29452,
  6253,
  33932,
  33814,
  558,
  46467,
  32222,
  31486,
  16440,
  37948,
  37800,
  42745,
  33872,
  10540],
 [44970,
  26924,
  40447,
  26577,
  27173,
  3566,
  39648,
  33814,
  660,
  40447,
  12752,
  26736,
  29452,
  10141,
  40551,
  48946,
  11750,
  8495,
  39455,
  33204,
  45008,
  26924,
  49826,
  9834,
  29452,
  8495,
  10027,
  26577,
  833,
  11902,
  29452],
 [3228,
  8134,
  46155,
  9284,
  4105,
  29458,
  12752,
  25448,
  48232,
  25252,
  29452,
  37853,
  29452,
  46155,
  49995,
  46155,
  15220,
  29452,
  26577,
  39942,
  8576,
  3114,
  9284,
  29452,
  25011,
  21950,
  40058,
  46155,
  18268]]

#Performed Padding
sent_len = max([len(sent) for sent in corpus])
embedded_docs = pad_sequences(one_hot_repr,padding="post",maxlen=sent_len)
embedded_docs[0]
     
array([29452, 12689,  3570, 37948, 46982, 27976,  6037,  3566, 21567,
       10695, 29452, 39105, 31282, 20722, 15501, 25448, 12639,  8530,
       43198, 11750,  8495, 30389, 35672, 12741, 31820,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0], dtype=int32)

#store the quality and difficulty in lists
rating = []
difficulty = []
for index,row in df.iterrows():
    #if type(row["comments"]) == str:
    rating.append(row["quality"])
    difficulty.append(row["difficulty"])

print(len(rating))
print(len(difficulty))
     
60993
60993

#Prepare the X and Y
X = np.array(embedded_docs)
Y = np.stack((rating,difficulty),axis=1)
     

#Y=np.array(Y,dtype='float32')
     

#Spliting the data with split ratio of 75-25%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(X,Y,test_size=0.25,random_state=42)

     

#build the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Dense, Flatten, GlobalMaxPooling1D, Embedding, Conv1D, LSTM, GRU, Bidirectional
model = Sequential([
    Embedding(voc_size,100,input_length=sent_len),
    Bidirectional(GRU(64,return_sequences=True)),
    Bidirectional(GRU(32)),
    Dropout(0.5),
    Dense(64,activation="relu"),
    Dense(32,activation="relu"),
    Dense(2,activation="linear")
])
model.summary()
     
Model: "sequential_12"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding_12 (Embedding)    (None, 411, 100)          5012200   
                                                                 
 bidirectional_16 (Bidirecti  (None, 411, 128)         63744     
 onal)                                                           
                                                                 
 bidirectional_17 (Bidirecti  (None, 64)               31104     
 onal)                                                           
                                                                 
 dropout_36 (Dropout)        (None, 64)                0         
                                                                 
 dense_38 (Dense)            (None, 64)                4160      
                                                                 
 dense_39 (Dense)            (None, 32)                2080      
                                                                 
 dense_40 (Dense)            (None, 2)                 66        
                                                                 
=================================================================
Total params: 5,113,354
Trainable params: 5,113,354
Non-trainable params: 0
_________________________________________________________________

#compile the model
model.compile(optimizer="Adam",loss="mse",metrics=['mae'])
     

pip install --upgrade tensorflow

     
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Requirement already satisfied: tensorflow in /usr/local/lib/python3.10/dist-packages (2.12.0)
Requirement already satisfied: packaging in /usr/local/lib/python3.10/dist-packages (from tensorflow) (23.1)
Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (0.32.0)
Requirement already satisfied: wrapt<1.15,>=1.11.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.14.1)
Requirement already satisfied: termcolor>=1.1.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (2.3.0)
Requirement already satisfied: six>=1.12.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.16.0)
Requirement already satisfied: google-pasta>=0.1.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (0.2.0)
Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (3.20.3)
Requirement already satisfied: tensorflow-estimator<2.13,>=2.12.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (2.12.0)
Requirement already satisfied: flatbuffers>=2.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (23.3.3)
Requirement already satisfied: astunparse>=1.6.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.6.3)
Requirement already satisfied: h5py>=2.9.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (3.8.0)
Requirement already satisfied: jax>=0.3.15 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (0.4.8)
Requirement already satisfied: gast<=0.4.0,>=0.2.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (0.4.0)
Requirement already satisfied: keras<2.13,>=2.12.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (2.12.0)
Requirement already satisfied: setuptools in /usr/local/lib/python3.10/dist-packages (from tensorflow) (67.7.2)
Requirement already satisfied: opt-einsum>=2.3.2 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (3.3.0)
Requirement already satisfied: absl-py>=1.0.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.4.0)
Requirement already satisfied: typing-extensions>=3.6.6 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (4.5.0)
Requirement already satisfied: tensorboard<2.13,>=2.12 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (2.12.2)
Requirement already satisfied: libclang>=13.0.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (16.0.0)
Requirement already satisfied: numpy<1.24,>=1.22 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.22.4)
Requirement already satisfied: grpcio<2.0,>=1.24.3 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.54.0)
Requirement already satisfied: wheel<1.0,>=0.23.0 in /usr/local/lib/python3.10/dist-packages (from astunparse>=1.6.0->tensorflow) (0.40.0)
Requirement already satisfied: ml-dtypes>=0.0.3 in /usr/local/lib/python3.10/dist-packages (from jax>=0.3.15->tensorflow) (0.1.0)
Requirement already satisfied: scipy>=1.7 in /usr/local/lib/python3.10/dist-packages (from jax>=0.3.15->tensorflow) (1.10.1)
Requirement already satisfied: requests<3,>=2.21.0 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.13,>=2.12->tensorflow) (2.27.1)
Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.13,>=2.12->tensorflow) (1.8.1)
Requirement already satisfied: google-auth-oauthlib<1.1,>=0.5 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.13,>=2.12->tensorflow) (1.0.0)
Requirement already satisfied: google-auth<3,>=1.6.3 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.13,>=2.12->tensorflow) (2.17.3)
Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.13,>=2.12->tensorflow) (3.4.3)
Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.13,>=2.12->tensorflow) (0.7.0)
Requirement already satisfied: werkzeug>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.13,>=2.12->tensorflow) (2.3.0)
Requirement already satisfied: cachetools<6.0,>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from google-auth<3,>=1.6.3->tensorboard<2.13,>=2.12->tensorflow) (5.3.0)
Requirement already satisfied: rsa<5,>=3.1.4 in /usr/local/lib/python3.10/dist-packages (from google-auth<3,>=1.6.3->tensorboard<2.13,>=2.12->tensorflow) (4.9)
Requirement already satisfied: pyasn1-modules>=0.2.1 in /usr/local/lib/python3.10/dist-packages (from google-auth<3,>=1.6.3->tensorboard<2.13,>=2.12->tensorflow) (0.3.0)
Requirement already satisfied: requests-oauthlib>=0.7.0 in /usr/local/lib/python3.10/dist-packages (from google-auth-oauthlib<1.1,>=0.5->tensorboard<2.13,>=2.12->tensorflow) (1.3.1)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorboard<2.13,>=2.12->tensorflow) (3.4)
Requirement already satisfied: charset-normalizer~=2.0.0 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorboard<2.13,>=2.12->tensorflow) (2.0.12)
Requirement already satisfied: urllib3<1.27,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorboard<2.13,>=2.12->tensorflow) (1.26.15)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorboard<2.13,>=2.12->tensorflow) (2022.12.7)
Requirement already satisfied: MarkupSafe>=2.1.1 in /usr/local/lib/python3.10/dist-packages (from werkzeug>=1.0.1->tensorboard<2.13,>=2.12->tensorflow) (2.1.2)
Requirement already satisfied: pyasn1<0.6.0,>=0.4.6 in /usr/local/lib/python3.10/dist-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<2.13,>=2.12->tensorflow) (0.5.0)
Requirement already satisfied: oauthlib>=3.0.0 in /usr/local/lib/python3.10/dist-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<1.1,>=0.5->tensorboard<2.13,>=2.12->tensorflow) (3.2.2)

#fit the model on training data
history = model.fit(x_train,y_train,validation_split=0.2,epochs=10,batch_size=128,verbose=1,shuffle=True)
     
Epoch 1/10
286/286 [==============================] - 77s 211ms/step - loss: 2.2472 - mae: 1.2060 - val_loss: 1.2104 - val_mae: 0.8944
Epoch 2/10
286/286 [==============================] - 29s 100ms/step - loss: 1.1031 - mae: 0.8378 - val_loss: 1.6533 - val_mae: 1.0806
Epoch 3/10
286/286 [==============================] - 24s 84ms/step - loss: 0.8593 - mae: 0.7258 - val_loss: 1.5856 - val_mae: 1.0624
Epoch 4/10
286/286 [==============================] - 21s 74ms/step - loss: 0.7095 - mae: 0.6480 - val_loss: 1.3974 - val_mae: 0.9837
Epoch 5/10
286/286 [==============================] - 24s 83ms/step - loss: 0.6113 - mae: 0.5911 - val_loss: 1.2892 - val_mae: 0.9419
Epoch 6/10
286/286 [==============================] - 25s 89ms/step - loss: 0.5415 - mae: 0.5492 - val_loss: 0.9434 - val_mae: 0.7716
Epoch 7/10
286/286 [==============================] - 21s 75ms/step - loss: 0.4923 - mae: 0.5181 - val_loss: 0.7917 - val_mae: 0.6801
Epoch 8/10
286/286 [==============================] - 21s 75ms/step - loss: 0.4568 - mae: 0.4935 - val_loss: 0.7345 - val_mae: 0.6461
Epoch 9/10
286/286 [==============================] - 20s 71ms/step - loss: 0.4237 - mae: 0.4708 - val_loss: 0.7148 - val_mae: 0.6213
Epoch 10/10
286/286 [==============================] - 21s 72ms/step - loss: 0.4001 - mae: 0.4530 - val_loss: 0.6895 - val_mae: 0.5989

# Plot the training and validation loss over each epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
     


#Predicted Labels
pred = model.predict(x_test)
pred[:5]
     
477/477 [==============================] - 12s 22ms/step
array([[1.1535242, 4.4296894],
       [4.6140027, 3.5580285],
       [2.9596906, 3.2839403],
       [4.6258373, 1.1433547],
       [3.9599059, 4.493196 ]], dtype=float32)

#Actual Labels and Predicted Labels comparison
print("Actual labels=")
print(y_test[:5])

print("Predicted labels=")
print(pred[:5])
     
Actual labels=
[[1. 5.]
 [5. 3.]
 [1. 5.]
 [2. 3.]
 [5. 5.]]
Predicted labels=
[[1.1535242 4.4296894]
 [4.6140027 3.5580285]
 [2.9596906 3.2839403]
 [4.6258373 1.1433547]
 [3.9599059 4.493196 ]]

#defined clear text function for inputs
def clear_text(s):
  text = remove_links(s)
  text = remove_email(text)
  text = remove_extra_spaces(text)
  text = remove_extra_line(text)
  text = expand_contractions(text)
  text = remove_digits(text)
  text = convert_lower(text)
  text = tokenize(text)
  text = [lm.lemmatize(word) for word in text if word not in stopwords.words('english')]
  text = " ".join(text)
  return text
     

#test case 1
s = input("Enter a comment = ")
s = clear_text(s)
one_hot_repr = [one_hot(s,voc_size)]
embedded_docs = pad_sequences(one_hot_repr,padding="post",maxlen=sent_len)
len(embedded_docs)
     
Enter a comment = Professor is really intelligent and best at his teaching and he teaches subject by making them easy
1

#result 1
rate,diff = model.predict(embedded_docs).flatten()
print("Quality = {}".format(round(rate)))
print("Difficulty = {}".format(round(diff)))
     
1/1 [==============================] - 0s 42ms/step
Quality = 5
Difficulty = 1

#test case 2
s = input("Enter a comment = ")
s = clear_text(s)
one_hot_repr = [one_hot(s,voc_size)]
embedded_docs = pad_sequences(one_hot_repr,padding="post",maxlen=sent_len)
len(embedded_docs)
     
Enter a comment =  professor's teaching is bad and unclear and his teaching is hard
1

#result 2
rate,diff = model.predict(embedded_docs).flatten()
print("Quality = {}".format(round(rate)))
print("Difficulty = {}".format(round(diff)))
     
1/1 [==============================] - 0s 57ms/step
Quality = 1
Difficulty = 5

#test case 3
s = input("Enter a comment = ")
s = clear_text(s)
one_hot_repr = [one_hot(s,voc_size)]
embedded_docs = pad_sequences(one_hot_repr,padding="post",maxlen=sent_len)
len(embedded_docs)
     
Enter a comment = He explains everything clearly.So subject seems to be okay.
1

#result 3
rate,diff = model.predict(embedded_docs).flatten()
print("Quality = {}".format(round(rate)))
print("Difficulty = {}".format(round(diff)))
     
1/1 [==============================] - 0s 35ms/step
Quality = 3
Difficulty = 3


