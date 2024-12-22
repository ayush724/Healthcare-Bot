# %%
import pandas as pd
import numpy as np

# %%
df = pd.read_csv('./dataset.csv')
#df.head()

# %%
for col in df.columns:   
    df[col] = df[col].str.replace('_',' ')
# df.head()   

# %%
symptoms= df.columns[1:18]
#print(symptoms)

# %%
df['All_Symptoms'] = df.apply(lambda row: ','.join(row[symptoms].dropna()), axis=1)
# df.head()

# %%
df.drop(symptoms, axis=1, inplace=True)


# %%
# df.head()

# %%
df1 = pd.read_csv('./symptom_precaution.csv')
# df1.head()


# %%
precautions= df1.columns[1:5]
# print(precautions)

# %%
df1['All_Precautions'] = df1.apply(lambda row: ','.join(row[precautions].dropna()), axis=1)
# df1.head()

# %%
df1.drop(precautions, axis=1, inplace=True)

# %%
# df1.head()

# %%
df2 = pd.merge(df ,df1 ,on='Disease',how='inner')

# %%
# df2.head()

# %%
def pre(x):
    return "The precautions to take: " + x

# %%
df2.All_Precautions = df2.All_Precautions.apply(pre)

# %%
# df2.head()

# %%
# df2.All_Precautions[0]

# %%
df2.All_Symptoms = df2.All_Symptoms.str.replace(',', ' ')

# %%
# df2.head()

# %%
# import nltk
# from nltk.tokenize import RegexpTokenizer
# import string

# %%
rg = df2.copy()

df3 = pd.read_csv('./symptom_Description.csv')
rg1 = df3.copy()
# %%
# the stop words for removing
victim = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
    'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't",
    'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn',
    "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
    'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',"mustn't", 'needn', "needn't", 'shan', "shan't",
    'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", "ff", "suffering",
    "I", "and"
])

# the Preprocessing function use to preprocess the text input and the train samples
# def preprocess_text(text):
#     text = text.lower()

#     text = text.translate(str.maketrans('', '', string.punctuation))

#     tokenizer = RegexpTokenizer(r'\w+')
#     tokens = tokenizer.tokenize(text)

#     tokens = [word for word in tokens if word not in victim]

#     return tokens 

# # %%
# from sklearn.feature_extraction.text import CountVectorizer
# cv = CountVectorizer(max_features=5000, stop_words="english")

# %% [markdown]
# 

# %%
# from sklearn.metrics.pairwise import cosine_similarity

# %%
# rg.head()

# %%
# rg.All_Symptoms = rg.All_Symptoms.apply(preprocess_text)

# %%
# rg.head()

# %%
# rg.All_Symptoms = rg.All_Symptoms.apply(lambda x:' '.join(map(str, x)))


# %%
# rg.head()

# %%
# vectors = cv.fit_transform(rg["All_Symptoms"])

# %%
# the unique features
# cv.get_feature_names()

# %%
# rg.head()

# %%
# data = df['All_Symptoms'].values # features
# labels = df['Disease'].values # target

# %%
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import f1_score, accuracy_score 
# from sklearn.preprocessing import LabelEncoder

# %%

# data = vectors.copy()
# print(data)

# %%
# x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size = 0.8,random_state=42)
# rnd_forest = RandomForestClassifier(random_state=42, max_features='sqrt', n_estimators= 500, max_depth=13)
# rnd_forest.fit(x_train,y_train)
# preds=rnd_forest.predict(x_test)
# print('F1-score% =', f1_score(y_test, preds, average='macro')*100, '|', 'Accuracy% =', accuracy_score(y_test, preds)*100)

# %%
# preprocessed_symptoms2 = preprocess_text("I have skin rash, nodal skin eruptions")
# preprocessed_symptoms2 = " ".join(preprocessed_symptoms2)
# user2 = cv.transform([preprocessed_symptoms2])
# predict1 = rnd_forest.predict(user2)

# %%
# print("Disease:", predict1)
# predict2=predict1[0]
# mat = rg[rg["Disease"] == predict2]
# dia = mat.All_Precautions.values[0]
# print("Precautions:" , dia)

# %%
# rg.head()

# %%
# def nlp_model(x):
#     preprocessed_symptoms = preprocess_text(x)
#     preprocessed_symptoms = " ".join(preprocessed_symptoms)
#     symptom_vector = cv.transform([preprocessed_symptoms])
#     res1 = rnd_forest.predict(symptom_vector)
#     res2=res1[0]
#     mat1 = rg[rg["Disease"] == res2]
#     dia1 = mat1.All_Precautions.values[0]
#     print("The Diagonisis Result: ")
#     print("Disease: ",res2)
#     print("Precautions:" , dia1)
    
#     inp = input("Do you wanna know anything about any disease(yes/no): ")
#     if inp == "yes":
#         dis = input("Enter the disease: ")
#         print("Your entered disease: ", dis)
#         mat = rg[rg["Disease"] == dis]
#         print("You may have suffered from the following symptoms: ")
#         sym = mat.All_Symptoms.values[0]
#         print(sym)
#         print("The Precautions of your disease are - ")
#         dia = mat.All_Precautions.values[0]
#         print(dia)
        
#     else:
#         print("Get well soon!!")

# # %%
# user = input("Enter Your Symptoms(atlest 3/4 symptoms needed): ")

# # %%
# nlp_model(user)


