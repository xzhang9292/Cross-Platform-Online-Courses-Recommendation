import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from collections import  Counter
plt.style.use('ggplot')
stop=set(stopwords.words('english'))
import re
from nltk.tokenize import word_tokenize
import gensim
import string
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM,Dense,SpatialDropout1D
from tensorflow.keras.initializers import Constant
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Flatten
import os
from tensorflow.keras import backend as K
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import string

#load job data
data = np.load('jobdetail.npy')
JobDescription = data[1]
Title = data[0]
JobRequirment = data[2]

#load course data
data = np.load('coursesinfo.npy')
courseDescription = data[:,1]
courseLearned = data[:,2]
courseTitle = data[:,0]
courseURL = data[:,3]
coursePlat = data[:,4]

#job clustering
embedding_dict={}
with open('glove.6B.200d.txt','r') as f:
    for line in f:
        values=line.split()
        word=values[0]
        vectors=np.asarray(values[1:],'float32')
        embedding_dict[word]=vectors
f.close()
MAX_LEN=100
tokenizer_obj=Tokenizer()
tokenizer_obj.fit_on_texts(JobDescription)
sequences=tokenizer_obj.texts_to_sequences(JobDescription)
job_description_pad=pad_sequences(sequences,maxlen=MAX_LEN,truncating='post',padding='post')
word_index=tokenizer_obj.word_index

num_words=len(word_index)+1
embedding_matrix=np.zeros((num_words,200))

for word,i in tqdm(word_index.items()):
    if i > num_words:
        continue
    
    emb_vec=embedding_dict.get(word)
    if emb_vec is not None:
        embedding_matrix[i]=emb_vec

model=Sequential()
embedding = Embedding(num_words, 200, weights=[embedding_matrix],trainable=False)
model.add(embedding)
model.add(SpatialDropout1D(0.2))
model.add(Flatten())
job_description_embedding = model.predict(job_description_pad)
db = DBSCAN(eps=1, min_samples=1).fit(job_description_pad)
dic = {}
for label in db.labels_:
    if label in dic:
        dic[label]+=1
    else:
        dic[label] = 1

sorted_jobs = [[k, v] for k, v in sorted(dic.items(), key=lambda item: item[1], reverse=True)]
cluster_to_job = {}
for i in range(len(db.labels_)):
    label = db.labels_[i]
    if label in cluster_to_job:
        cluster_to_job[label].append(i)
    else:
        cluster_to_job[label] = [i]

#extract key words from job 
import pke
def extract_keyphrases(caption, n):
    extractor = pke.unsupervised.TextRank() 
    extractor.load_document(caption)
    extractor.candidate_selection()
    extractor.candidate_weighting()
    keyphrases = extractor.get_n_best(n=n, stemming=False)
    return(keyphrases)
def keywords_from_cluster(label):
    indexlist = cluster_to_job[label]
    keyskills = {}
    for index in indexlist:
        requirement = JobRequirment.iloc[index]
        keywords = [i[0] for i in extract_keyphrases(requirement,5)]
        keywords.append(Title.iloc[index])
        for w in keywords:
            if w in keyskills:
                keyskills[w] += 1
            else:
                keyskills[w] = 1
    out = [[k,keyskills[k]] for k in keyskills]
    out = sorted(out, key=second,reverse = True)
    topkey = [k[0] for k in out[:10]]

    return topkey
def second(v):
    return v[1]

cluster_to_keyword = {}
for label in cluster_to_job:
    keyset = keywords_from_cluster(label)
    
    cluster_to_keyword[label] = keyset

# extract key words from course
course_to_keyword = {}
for c in range(len(courseLearned)):
    title = courseTitle[c]
    if not title in course_to_keyword:
        course_to_keyword[title] = [v[0] for v in extract_keyphrases(str(courseDescription[c]),5)]
        course_to_keyword[title].append(title)

#Job Cluster and Keyword matching
def related_score(key_skill, key_learn):
    token = key_skill.split()
    total = len(token)
    count = 0
    for i in token:
        if i in key_learn:
            count+=1
    return count/total

def skillsets_to_course(skillsets):
    scoreboard = []
    
    skill = ' '.join(skillsets)
    title = []
    content = []
    for i in course_to_keyword:
        coursekey = course_to_keyword[i]
        title.append(i)
        content.append(' '.join(coursekey))
    content.append(skill)
    vec = TfidfVectorizer()
    response = vec.fit_transform(content)
    out = csr_matrix(response).toarray()
    for index in range(len(out) - 1):
        scoreboard.append([title[index],cosine_similarity([out[index]],[out[-1]])])
    output = [v[0] for v in sorted(scoreboard, key=lambda item: item[1], reverse = True)]
    return output[:100]

def check_if_english(title):
    
    for i in title.split():
        if not isEnglish(i):
            return False
    return True

def isEnglish(s):
    return s.isascii()

course_matching_map = {}
for i in cluster_to_keyword:
    out = skillsets_to_course(cluster_to_keyword[i])
    #clean duplicate:
    out_cleaned = []
    outset = set()
    for t in out:
        if (not t.lower() in outset) and check_if_english(t):
            outset.add(t.lower())
            out_cleaned.append(t)
    course_matching_map[i] = out_cleaned

# Save to json
import json
course_matching_map_out = {}
for i in course_matching_map:
    course_matching_map_out[int(i)] = course_matching_map[i]
with open('course_matching_map_out_all_plat.json', 'w') as f:
        json.dump(course_matching_map_out, f)
joblabels = {}
for i in range(len(db.labels_)):
    joblabels[i] = int(db.labels_[i])
with open('joblabel.json', 'w') as f:
        json.dump(joblabels, f)
with open('joblabel.json', 'r') as f:
        check = json.load(f)
jobtitle = {}
for i in range(len(titles)):
    jobtitle[i] = titles.iloc[i]
with open('title.json', 'w') as f:
        json.dump(jobtitle, f)
url_out = {}
for i in range(len(courseTitle)):
    url_out[courseTitle[i]] = courseURL[i]
with open('url_all_plat.json', 'w') as f:
        json.dump(url_out, f)
plat_out = {}
for i in range(len(courseTitle)):
    plat_out[courseTitle[i]] = coursePlat[i]
with open('platform_all_plat.json', 'w') as f:
        json.dump(plat_out, f)


