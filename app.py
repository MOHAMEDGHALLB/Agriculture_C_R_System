from flask import Flask, render_template, request, redirect, url_for
from flask_mysqldb import MySQL
import numpy as np
import joblib

# Load models and scalers
model = joblib.load('C:/Users/MTechno/Desktop/D_A_P/python/crop_Monday/Crop-Recommendation-System-Using-Machine-Learning/model')
sc = joblib.load('C:/Users/MTechno/Desktop/D_A_P/python/crop_Monday/Crop-Recommendation-System-Using-Machine-Learning/standscaler')
ms = joblib.load('C:/Users/MTechno/Desktop/D_A_P/python/crop_Monday/Crop-Recommendation-System-Using-Machine-Learning/minmaxscaler')

app = Flask(__name__, template_folder='templates')

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root'
app.config['MYSQL_DB'] = 'crop'

mysql = MySQL(app)

@app.route('/')
@app.route('/home')
def home():
    return render_template("home.html")

@app.route('/About_Project')
def about_project():
    return render_template("about_Project.html")



def extract_form_data():
    form_data = request.form
    N = form_data['Nitrogen']
    P = form_data['Phosphorus']
    K = form_data['Potassium']
    temp = form_data['Température']
    humidity = form_data['Humidité']
    ph = form_data['ph']
    rainfall = form_data['pluviométrie']
    return N, P, K, temp, humidity, ph, rainfall

@app.route('/data', methods=['GET', 'POST'])
def data():
    if request.method == 'POST':
        try:
            N, P, K, temp, humidity, ph, rainfall = extract_form_data()
            date = request.form['date']
            city = request.form['Ville']

            cursor = mysql.connection.cursor()
            sql = """INSERT INTO data_crop (Nitrogen, phosphorus, K, temperature, humidity, ph, pluviométrie, date, city)
                     VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"""
            cursor.execute(sql, (N, P, K, temp, humidity, ph, rainfall, date, city))
            mysql.connection.commit()
            cursor.close()

            feature_list = [N, P, K, temp, humidity, ph, rainfall]
            single_pred = np.array(feature_list).reshape(1, -1)

            scaled_features = ms.transform(single_pred)
            final_features = sc.transform(scaled_features)
            prediction = model.predict(final_features)

            crop_dict = {
                1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
            }

            if prediction[0] in crop_dict:
                crop = crop_dict[prediction[0]]
                result = f"Based on the data you have provided, {crop} appears to be the most suitable crop for this area."
            else:
                result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
# when the data successfully insert into database will this message show
            return render_template("home.html", message="Data inserted successfully!", result=result)
        except Exception as e:
            return render_template("home.html", message="An error occurred: " + str(e))

    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)



