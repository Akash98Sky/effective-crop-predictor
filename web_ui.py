from flask import Flask, request, jsonify, render_template, session, url_for, redirect
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from wtforms import TextField, SubmitField, validators
from flask_wtf import FlaskForm

crop_imgs = {
    'Cotton': 'https://images.unsplash.com/photo-1502395809857-fd80069897d0?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1170&q=80',
    'Sugar Cane': 'https://images.unsplash.com/photo-1605810978644-1fa3b63ecd4e?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=687&q=80',
    'Jowar': 'https://media.istockphoto.com/photos/ripe-sorghum-milo-millet-crop-field-in-rows-picture-id503285337',
    'Bajra': 'https://media.istockphoto.com/photos/millet-crop-picture-id846255882',
    'Soyabeans': 'https://media.istockphoto.com/photos/soya-beans-picture-id171309749',
    'Corn': 'https://images.unsplash.com/photo-1511817354854-e361703ac368?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=1167&q=80',
    'Rice': 'https://images.unsplash.com/photo-1600387845879-a4713f764110?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=1170&q=80',
    'Wheat': 'https://images.unsplash.com/photo-1594842059196-5b6b8fa73187?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=1074&q=80',
    'Ground Nut': 'https://media.istockphoto.com/photos/walnuts-with-hard-shell-in-wicker-basket-picture-id1182307929'
}

def return_prediction(model, scaler, sample_json):

    ph = sample_json["pH"]
    n = sample_json["N"]
    p = sample_json["P"]
    k = sample_json["K"]
    temp = sample_json["Temp"]
    rain = sample_json["Rain"]

    crop = [[ph, n, p, k, temp, rain]]

    classes = np.array(['Cotton', 'Sugar Cane', 'Jowar', 'Bajra',
                        'Soyabeans', 'Corn', 'Rice', 'Wheat', 'Ground Nut'])

    crop = scaler.transform(crop)

    class_ind = model.predict_classes(crop)[0]

    return classes[class_ind]


app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'


class CropForm(FlaskForm):

    ph_len = TextField("pH", [validators.DataRequired()])
    n_len = TextField("N", [validators.DataRequired()])
    p_len = TextField("P", [validators.DataRequired()])
    k_len = TextField("K",  [validators.DataRequired()])
    temp_len = TextField("Temperature", [validators.DataRequired()])
    rain_len = TextField("Rain", [validators.DataRequired()])

    submit = SubmitField("Analyze")

def validateForm():
    pass

@app.route("/", methods=['GET', 'POST'])
def index():
    form = CropForm()

    if request.method == 'POST' and form.validate():
        session['ph_len'] = form.ph_len.data
        session['n_len'] = form.n_len.data
        session['p_len'] = form.p_len.data
        session['k_len'] = form.k_len.data
        session['temp_len'] = form.temp_len.data
        session['rain_len'] = form.rain_len.data

        return redirect(url_for("prediction"))
    return render_template('home.html', form=form)


crop_model = load_model("./saved/crop_model.h5")
crop_scaler = joblib.load("./saved/crop_scaler.pkl")


@app.route('/prediction')
def prediction():
    content = {}

    content['pH'] = float(session['ph_len'])
    content['N'] = float(session['n_len'])
    content['P'] = float(session['p_len'])
    content['K'] = float(session['k_len'])
    content['Temp'] = float(session['temp_len'])
    content['Rain'] = float(session['rain_len'])

    prediction = return_prediction(crop_model, crop_scaler, content)
    results = {
        'name': prediction,
        'image': crop_imgs[prediction]
    }

    return render_template('prediction.html', results=results)


if __name__ == '__main__':
    app.run()
