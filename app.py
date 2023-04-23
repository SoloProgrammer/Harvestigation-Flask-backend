from flask import Flask, request, jsonify
# from tkinter import *
# Importing pyttsx3 library to convert text into speech.
# import pyttsx3
# Importing pandas library
import pandas as pd
# Importing sklearn library. This is a very powerfull library for machine learning. Scikit-learn is probably the most useful library for machine learning in Python. The sklearn library contains a lot of efficient tools for machine learning and statistical modeling including classification, regression, clustering and dimensionality reduction.
from sklearn import preprocessing
# Importing Knn Classifier from sklearn library.
from sklearn.neighbors import KNeighborsClassifier
# Importing numpy to do stuffs related to arrays
import numpy as np
# Importing pysimplegui to make a Graphical User Interface.
# import PySimpleGUI as sg
from flask_cors import CORS
import os


def PredictCropWithReact(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    # Importing our excel data from a specific file.
    excel = pd.read_excel('Crop.xlsx', header=0)
    # Various machine learning algorithms require numerical input data, so you need to represent categorical columns in a numerical column. In order to encode this data, you could map each value to a number. This process is known as label encoding, and sklearn conveniently will do this for you using Label Encoder.
    le = preprocessing.LabelEncoder()
    # Mapping the values in weather into numerical form.
    crop = le.fit_transform(list(excel["CROP"]))

    # Making the whole row consisting of nitrogen values to come into nitrogen.
    NITROGEN = list(excel["NITROGEN"])
    # Making the whole row consisting of phosphorus values to come into phosphorus.
    PHOSPHORUS = list(excel["PHOSPHORUS"])
    # Making the whole row consisting of potassium values to come into potassium.
    POTASSIUM = list(excel["POTASSIUM"])
    # Making the whole row consisting of temperature values to come into temperature.
    TEMPERATURE = list(excel["TEMPERATURE"])
    # Making the whole row consisting of humidity values to come into humidity.
    HUMIDITY = list(excel["HUMIDITY"])
    # Making the whole row consisting of ph values to come into ph.
    PH = list(excel["PH"])
    RAINFALL = list(excel["RAINFALL"])

    # Zipping all the features together
    features = list(zip(NITROGEN, PHOSPHORUS, POTASSIUM,
                    TEMPERATURE, HUMIDITY, PH, RAINFALL))
    # Converting all the features into a array form
    features = np.array([NITROGEN, PHOSPHORUS, POTASSIUM,
                        TEMPERATURE, HUMIDITY, PH, RAINFALL])

    # Making transpose of the features
    features = features.transpose()
    # Printing the shape of the features after getting transposed.
    print(features.shape)
    # Printing the shape of crop. Please note that the shape of the features and crop should match each other to make predictions.
    print(crop.shape)

    # The number of neighbors is the core deciding factor. K is generally an odd number if the number of classes is 2. When K=1, then the algorithm is known as the nearest neighbor algorithm.
    model = KNeighborsClassifier(n_neighbors=3)
    # Making the whole row consisting of rainfall values to come into rainfall.
    model.fit(features, crop)
    # Taking input from the user about nitrogen content in the soil.
    nitrogen_content = int(nitrogen)
    # Taking input from the user about phosphorus content in the soil.
    phosphorus_content = int(phosphorus)
    # Taking input from the user about potassium content in the soil.
    potassium_content = int(potassium)
    # Taking input from the user about the surrounding temperature.
    temperature_content = int(temperature)
    # Taking input from the user about the surrounding humidity.
    humidity_content = int(humidity)
    # Taking input from the user about the ph level of the soil.
    ph_content = int(ph)
    # Taking input from the user about the rainfall.
    rainfall = int(rainfall)
    # Converting all the data that we collected from the user into a array form to make further predictions.
    predict1 = np.array([nitrogen_content, phosphorus_content, potassium_content,
                        temperature_content, humidity_content, ph_content, rainfall])
    # Printing the data after being converted into a array form.
    print(predict1)
    # Reshaping the input data so that it can be applied in the model for getting accurate results.
    predict1 = predict1.reshape(1, -1)
    # Printing the input data value after being reshaped.
    print(predict1)
    # Applying the user input data into the model.
    predict1 = model.predict(predict1)
    # Finally printing out the results.
    print(predict1)
    crop_name = str()
    # Above we have converted the crop names into numerical form, so that we can apply the machine learning model easily. Now we have to again change the numerical values into names of crop so that we can print it when required.
    if predict1 == 0:
        crop_name = 'Apple(सेब)'
    elif predict1 == 1:
        crop_name = 'Banana(केला)'
    elif predict1 == 2:
        crop_name = 'Blackgram(काला चना)'
    elif predict1 == 3:
        crop_name = 'Chickpea(काबुली चना)'
    elif predict1 == 4:
        crop_name = 'Coconut(नारियल)'
    elif predict1 == 5:
        crop_name = 'Coffee(कॉफ़ी)'
    elif predict1 == 6:
        crop_name = 'Cotton(कपास)'
    elif predict1 == 7:
        crop_name = 'Grapes(अंगूर)'
    elif predict1 == 8:
        crop_name = 'Jute(जूट)'
    elif predict1 == 9:
        crop_name = 'Kidneybeans(राज़में)'
    elif predict1 == 10:
        crop_name = 'Lentil(मसूर की दाल)'
    elif predict1 == 11:
        crop_name = 'Maize(मक्का)'
    elif predict1 == 12:
        crop_name = 'Mango(आम)'
    elif predict1 == 13:
        crop_name = 'Mothbeans(मोठबीन)'
    elif predict1 == 14:
        crop_name = 'Mungbeans(मूंग)'
    elif predict1 == 15:
        crop_name = 'Muskmelon(खरबूजा)'
    elif predict1 == 16:
        crop_name = 'Orange(संतरा)'
    elif predict1 == 17:
        crop_name = 'Papaya(पपीता)'
    elif predict1 == 18:
        crop_name = 'Pigeonpeas(कबूतर के मटर)'
    elif predict1 == 19:
        crop_name = 'Pomegranate(अनार)'
    elif predict1 == 20:
        crop_name = 'Rice(चावल)'
    elif predict1 == 21:
        crop_name = 'Watermelon(तरबूज)'

    # Here I have divided the humidity values into three categories i.e low humid, medium humid, high humid.
    if int(humidity_content) >= 1 and int(humidity_content) <= 33:
        humidity_level = 'low humid'
    elif int(humidity_content) >= 34 and int(humidity_content) <= 66:
        humidity_level = 'medium humid'
    else:
        humidity_level = 'high humid'

    # Here I have divided the temperature values into three categories i.e cool, warm, hot.
    if int(temperature_content) >= 0 and int(temperature_content) <= 6:
        temperature_level = 'cool'
    elif int(temperature_content) >= 7 and int(temperature_content) <= 25:
        temperature_level = 'warm'
    else:
        temperature_level = 'hot'

    # Here I have divided the humidity values into three categories i.e less, moderate, heavy rain.
    if int(rainfall) >= 1 and int(rainfall) <= 100:
        rainfall_level = 'less'
    elif int(rainfall) >= 101 and int(rainfall) <= 200:
        rainfall_level = 'moderate'
    elif int(rainfall) >= 201:
        rainfall_level = 'heavy rain'

    # Here I have divided the nitrogen values into three categories.
    if int(nitrogen_content) >= 1 and int(nitrogen_content) <= 50:
        nitrogen_level = 'less'
    elif int(nitrogen_content) >= 51 and int(nitrogen_content) <= 100:
        nitrogen_level = 'not to less but also not to high'
    elif int(nitrogen_content) >= 101:
        nitrogen_level = 'high'

    # Here I have divided the phosphorus values into three categories.
    if int(phosphorus_content) >= 1 and int(phosphorus_content) <= 50:
        phosphorus_level = 'less'
    elif int(phosphorus_content) >= 51 and int(phosphorus_content) <= 100:
        phosphorus_level = 'not to less but also not to high'
    elif int(phosphorus_content) >= 101:
        phosphorus_level = 'high'

    # Here I have divided the potassium values into three categories.
    if int(potassium_content) >= 1 and int(potassium_content) <= 50:
        potassium_level = 'less'
    elif int(potassium_content) >= 51 and int(potassium_content) <= 100:
        potassium_level = 'not to less but also not to high'
    elif int(potassium_content) >= 101:
        potassium_level = 'high'

    # Here I have divided the ph values into three categories.
    if float(ph_content) >= 0 and float(ph_content) <= 5:
        phlevel = 'acidic'
    elif float(ph_content) >= 6 and float(ph_content) <= 8:
        phlevel = 'neutral'
    elif float(ph_content) >= 9 and float(ph_content) <= 14:
        phlevel = 'alkaline'

    print(crop_name)

    return {"detail": "<p> <h6>Sir according to the data that you provided to us.</h6> <ul> <li>The ratio of <u>nitrogen</u> in the soil is  " + "<i>" + nitrogen_level + "</i>" + ".</li>" + " \n <li>The ratio of <u>phosphorus</u> in the soil is  " + "<i>" + phosphorus_level + "</i>" + ".</li>" + "<li> The ratio of <u>potassium</u> in the soil is  " + "<i>" + potassium_level + "</i>" + ".</li>" + "<li> The <u>temperature</u> level around the field is " + "<i>" +
            temperature_level + "</i>" + ".</li> " + "<li>The <u>humidity</u> level around the field is  " + "<i>" + humidity_level + "</i>" + ".</li>" + "<li>The <u>ph</u> type of the soil is  " + "<i>" + phlevel + "</i>" + ".</li>" + "<li> The amount of <u>rainfall</u> is  " + "<i>" + rainfall_level + "</i>" + ".</li>" + "</ul></p>", "crop_name": crop_name}


# Creating our flask app
app = Flask(__name__)

# Cross Origin Resource Sharing Setup
CORS(app)

# Defining POST route


@app.route('/')
def start():
    return jsonify({"message":"Server is running.."})

@app.route('/harvest', methods=["POST"])
def harvest():

    request_data = request.json['soilData']

    print(request_data)
    nitrogen = request_data['nitrogen']
    phosphorus = request_data['phosphorus']
    potassium = request_data['potassium']
    temperature = request_data['temperature']
    humidity = request_data['humidity']
    ph = request_data['ph']
    rainfall = request_data['rainfall']

    return PredictCropWithReact(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)


if __name__ == "__main__":
    app.run(debug=True, port=os.getenv("PORT", default=5000))
