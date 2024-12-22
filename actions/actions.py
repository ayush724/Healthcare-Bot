from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import string
#for date and time
import datetime as dt
#for disease description
import wikipediaapi
import wikipedia
#for hospital location
from geopy.geocoders import Nominatim
import requests
from flask import Flask,render_template


# import  disease_prediction_python
from . import disease_prediction_python
rg = disease_prediction_python.rg
rg1 = disease_prediction_python.rg1


# portion for preprocess_text code
# the stop words for removing
victim = {
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
            'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
            'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", "ff",
            "suffering", "I", "and"
        }


# the Preprocessing function use to preprocess the text input and the train samples
def preprocess_text(text):
    text = text.lower()

    text = text.translate(str.maketrans('', '', string.punctuation))

    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)

    tokens = [word for word in tokens if word not in victim]


    return tokens





class ActionGetDiagnosis(Action):
    def name(self) -> Text:
        return "action_get_diagnosis"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Get user input from the tracker
        user_input = tracker.latest_message.get("text")

        # Preprocess user input
        preprocessed_symptoms = preprocess_text(user_input)
        preprocessed_symptoms = " ".join(preprocessed_symptoms)

        # from sklearn.feature_extraction.text import CountVectorizer
        # cv = CountVectorizer(max_features=5000, stop_words="english")

        from sklearn.feature_extraction.text import TfidfVectorizer
        cv1 = TfidfVectorizer()

        vectors = cv1.fit_transform(rg["All_Symptoms"])
        
        # using random forest 
        preprocessed_symptoms = cv1.transform([preprocessed_symptoms])
        data = vectors.copy()
        labels = rg['Disease'].values
        x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size = 0.8,random_state=42)
        rnd_forest = RandomForestClassifier(random_state=42, max_features='sqrt', n_estimators= 500, max_depth=13)
        rnd_forest.fit(x_train,y_train)
        res1 = rnd_forest.predict(preprocessed_symptoms)
        disease = res1[0]
        mat1 = rg[rg["Disease"] == disease]
        diagnosis = mat1.All_Precautions.values[0]
        # for disease description
        # mat2 = rg1[rg1["Disease"] == disease]
        # description = mat2.Description.values[0]
        
        # using cosine cosine_similarity
        # symptom_vector = cv.transform([preprocessed_symptoms])
        # similar = cosine_similarity(vectors, symptom_vector)
        # index = similar.argmax()
        # diagnosis = rg.loc[index, "All_Precautions"]
        # disease = rg.loc[index, "Disease"]
  
        if disease=="Paralysis (brain hemorrhage)":
            disease="Paralysis"
        elif disease =="(vertigo) Paroymsal  Positional Vertigo":
            disease="Vertigo"
        elif disease =="Dimorphic hemmorhoids(piles)":
            disease="Hemorrhoids"
        description = wikipedia.summary(disease, auto_suggest=False,sentences=4)
        # Send the diagnosis as a response to the user
        dispatcher.utter_message(text=f"According to your symptoms, you may have {disease}.  \n"
                                 f"Description: {description}  \n"
                                      f"{diagnosis}.  \n Did you find this helpful?")

        return []

class ActionDiseaseInfo(Action):
    def name(self) -> Text:
        return "action_disease_info"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        disease = tracker.get_slot("disease")
       
        try:
          info = wikipedia.summary(disease, auto_suggest=False,sentences=4)
          dispatcher.utter_message(
                text=f"Here is some information about {disease} => \n{info} || Is this helpful ?"
            )
        except  wikipedia.exceptions.DisambiguationError:
          dispatcher.utter_message(
                text=f"Multile pages found please be more specific with the detail"
            )
        except wikipedia.exceptions.PageError:
          dispatcher.utter_message(
                text= "I'm sorry, I couldn't find information about that disease. Please try a different "
                     "disease or use the disease with a correct spelling"
            )


        return []
    

class ActionGettime(Action):


    def name(self) -> Text:
        return "action_give_time"

    def run(self, dispatcher: CollectingDispatcher,
             tracker: Tracker,
             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:


        # Send the diagnosis as a response to the user
        dispatcher.utter_message(text=f"The current time is - {dt.datetime.now()}")

        return []
    
# class ActionGetLocation(Action):
#     def name(self) -> Text:
#         return "action_get_location"

#     def run(
#         self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
#     ) -> List[Dict[Text, Any]]:
        # location = tracker.latest_message.get("text")
        #   location = preprocess_text(location)
#         location = tracker.get_slot("location")
#         # calling the Nominatim tool
#         loc = Nominatim(user_agent="GetLoc")
#         getLoc = loc.geocode(location)
#         latitude = getLoc.latitude
#         longitude = getLoc.longitude

#         URL = "https://discover.search.hereapi.com/v1/discover"

#         api_key = 'xz4AIKWkkBwQJsb0kY_nas3XHBqXjLcrSydMRhJ6J7U' # Acquire from developer.here.com
#         query = 'hospital'
#         limit = 5

#         PARAMS = {
#             'apikey':api_key,
#             'q':query,
#             'limit': limit,
#             'at':'{},{}'.format(latitude,longitude)
#          } 

#         # sending get request and saving the response as response object 
#         r = requests.get(url = URL, params = PARAMS) 
#         data = r.json()


#         hospitalOne = data['items'][0]['title']
#         hospitalOne_address =  data['items'][0]['address']['label']
#         hospitalOne_latitude = data['items'][0]['position']['lat']
#         hospitalOne_longitude = data['items'][0]['position']['lng']


#         hospitalTwo = data['items'][1]['title']
#         hospitalTwo_address =  data['items'][1]['address']['label']
#         hospitalTwo_latitude = data['items'][1]['position']['lat']
#         hospitalTwo_longitude = data['items'][1]['position']['lng']

#         hospitalThree = data['items'][2]['title']
#         hospitalThree_address =  data['items'][2]['address']['label']
#         hospitalThree_latitude = data['items'][2]['position']['lat']
#         hospitalThree_longitude = data['items'][2]['position']['lng']


#         hospitalFour = data['items'][3]['title']
#         hospitalFour_address =  data['items'][3]['address']['label']
#         hospitalFour_latitude = data['items'][3]['position']['lat']
#         hospitalFour_longitude = data['items'][3]['position']['lng']

#         hospitalFive = data['items'][4]['title']
#         hospitalFive_address =  data['items'][4]['address']['label']
#         hospitalFive_latitude = data['items'][4]['position']['lat']
#         hospitalFive_longitude = data['items'][4]['position']['lng']

#         app = Flask(__name__,template_folder='nearby_hospitals')
#         @app.route('/')

#         def map_func():
#             return render_template('hospital.html',
#                                      latitude = latitude,
#                                      longitude = longitude,
#                                      apikey=api_key,
#                                      oneName=hospitalOne,
#                                      OneAddress=hospitalOne_address,
#                                      oneLatitude=hospitalOne_latitude,
#                                      oneLongitude=hospitalOne_longitude,
#                                      twoName=hospitalTwo,
#                                      twoAddress=hospitalTwo_address,
#                                      twoLatitude=hospitalTwo_latitude,
#                                      twoLongitude=hospitalTwo_longitude,
#                                      threeName=hospitalThree,
#                                      threeAddress=hospitalThree_address,
#                                      threeLatitude=hospitalThree_latitude,
#                                      threeLongitude=hospitalThree_longitude,
#                                      fourName=hospitalFour,		
#                                      fourAddress=hospitalFour_address,
#                                      fourLatitude=hospitalFour_latitude,
#                                      fourLongitude=hospitalFour_longitude,
#                                      fiveName=hospitalFive,		
#                                      fiveAddress=hospitalFive_address,
#                                      fiveLatitude=hospitalFive_latitude,
#                                      fiveLongitude=hospitalFive_longitude
#                             )

#         if __name__ == '__main__':
# 	        app.run(debug = True)

#         dispatcher.utter_message(text=f"The current time is - ")

#         return []


