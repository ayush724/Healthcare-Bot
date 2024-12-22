# api id: 8DPAMcSxV0J7jTbnqIsI
from geopy.geocoders import Nominatim
import requests
from flask import Flask,render_template,request,redirect,url_for
URL = "https://discover.search.hereapi.com/v1/discover"

api_key = 'xz4AIKWkkBwQJsb0kY_nas3XHBqXjLcrSydMRhJ6J7U' 
query = 'hospital'
limit = 5

app = Flask(__name__,template_folder='nearby_hospitals')

@app.route('/')
def my_form():
    return render_template('take_location.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    location = text.upper()
    loc = Nominatim(user_agent="GetLoc")
    getLoc = loc.geocode(location)
    latitude1=getLoc.latitude
    longitude1=getLoc.longitude
   
    PARAMS = {
            'apikey':api_key,
            'q':query,
            'limit': limit,
            'at':'{},{}'.format(latitude1,longitude1)
         } 

    # sending get request and saving the response as response object 
    r = requests.get(url = URL, params = PARAMS) 
    data = r.json()


    hospitalOne = data['items'][0]['title']
    hospitalOne_address =  data['items'][0]['address']['label']
    hospitalOne_latitude = data['items'][0]['position']['lat']
    hospitalOne_longitude = data['items'][0]['position']['lng']


    hospitalTwo = data['items'][1]['title']
    hospitalTwo_address =  data['items'][1]['address']['label']
    hospitalTwo_latitude = data['items'][1]['position']['lat']
    hospitalTwo_longitude = data['items'][1]['position']['lng']

    hospitalThree = data['items'][2]['title']
    hospitalThree_address =  data['items'][2]['address']['label']
    hospitalThree_latitude = data['items'][2]['position']['lat']
    hospitalThree_longitude = data['items'][2]['position']['lng']


    hospitalFour = data['items'][3]['title']
    hospitalFour_address =  data['items'][3]['address']['label']
    hospitalFour_latitude = data['items'][3]['position']['lat']
    hospitalFour_longitude = data['items'][3]['position']['lng']

    hospitalFive = data['items'][4]['title']
    hospitalFive_address =  data['items'][4]['address']['label']
    hospitalFive_latitude = data['items'][4]['position']['lat']
    hospitalFive_longitude = data['items'][4]['position']['lng']

    return render_template('hospital.html',
                            latitude = latitude1,
                            longitude = longitude1,
                            apikey=api_key,
                            oneName=hospitalOne,
                            OneAddress=hospitalOne_address,
                            oneLatitude=hospitalOne_latitude,
                            oneLongitude=hospitalOne_longitude,
                            twoName=hospitalTwo,
                            twoAddress=hospitalTwo_address,
                            twoLatitude=hospitalTwo_latitude,
                            twoLongitude=hospitalTwo_longitude,
                            threeName=hospitalThree,
                            threeAddress=hospitalThree_address,
                            threeLatitude=hospitalThree_latitude,
                            threeLongitude=hospitalThree_longitude,
                            fourName=hospitalFour,		
                            fourAddress=hospitalFour_address,
                            fourLatitude=hospitalFour_latitude,
                            fourLongitude=hospitalFour_longitude,
                            fiveName=hospitalFive,		
                            fiveAddress=hospitalFive_address,
                            fiveLatitude=hospitalFive_latitude,
                            fiveLongitude=hospitalFive_longitude
                            )


if __name__ == '__main__':
	app.run(port=8001,debug = True)
















# @app.route('/mapfunc',methods=['GET', 'POST'])
# def map_func():
#    # location_data = my_form_post()
#    # latitude = location_data.latitude
#    # longitude = location_data.longitude
#    return render_template('hospital.html',
#                             latitude = latitude,
#                             longitude = longitude,
#                             apikey=api_key,
#                             oneName=hospitalOne,
#                             OneAddress=hospitalOne_address,
#                             oneLatitude=hospitalOne_latitude,
#                             oneLongitude=hospitalOne_longitude,
#                             twoName=hospitalTwo,
#                             twoAddress=hospitalTwo_address,
#                             twoLatitude=hospitalTwo_latitude,
#                             twoLongitude=hospitalTwo_longitude,
#                             threeName=hospitalThree,
#                             threeAddress=hospitalThree_address,
#                             threeLatitude=hospitalThree_latitude,
#                             threeLongitude=hospitalThree_longitude,
#                             fourName=hospitalFour,		
#                             fourAddress=hospitalFour_address,
#                             fourLatitude=hospitalFour_latitude,
#                             fourLongitude=hospitalFour_longitude,
#                             fiveName=hospitalFive,		
#                             fiveAddress=hospitalFive_address,
#                             fiveLatitude=hospitalFive_latitude,
#                             fiveLongitude=hospitalFive_longitude
#                             )
