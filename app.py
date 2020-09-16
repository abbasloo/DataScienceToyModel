from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
import sqlite3
import keras
import numpy as np
import io
import pandas as pd
from flask import request
from flask import Flask
from flask_cors import CORS, cross_origin
from sklearn import datasets, linear_model
from focal_loss import BinaryFocalLoss
import pickle
from flask import Flask,render_template
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.offline as plt
import plotly
import json

regr = linear_model.LinearRegression()
mean = np.array([ 1.97044292e-08,  1.49070225e-08,  8.51945815e-09,  1.32319914e-08,
  3.34574985e-01,  3.05782514e-01, -4.74801265e-01,  1.11355449e-02,
  2.71881521e-03,  1.81857239e-03,  1.54335892e-02, -2.89997385e-01,
  4.72228523e-03,  5.95005891e-03,  1.38233034e-03,  2.29392246e+00,
  2.76877304e+00,  1.59555991e-08,  1.07062256e-08,  8.06360171e-09,
  4.02986296e-09,  2.12756661e-09,  6.00992003e-09,  1.09547534e-08,
 -2.55694733e-01,  9.57021663e-03,  2.35215256e-03,  1.62525501e-03,
 -1.45467999e-01,  1.44413623e-02,  4.12537209e-03,  5.74863430e-03,
  1.01508013e-08,  2.14347898e-09,  7.73720359e-09,  5.32951138e-09])
std = np.array([1.05561553e-07, 1.05906706e-07, 4.76488092e-08, 5.56488675e-08,
 1.74799965e-01, 2.48729980e-01, 1.31033408e+00, 3.29613492e-02,
 9.53623761e-03, 4.27176315e-03, 2.73628426e-02, 1.30049499e+00,
 1.18951669e-02, 9.95170699e-03, 5.15187911e-03, 5.59537537e+00,
 4.08057187e+00, 9.81827629e-08, 6.43430234e-08, 4.70108873e-08,
 1.73067360e-08, 1.47014394e-08, 2.92390469e-08, 3.83704023e-08,
 9.58850856e-01, 3.14805764e-02, 9.53379485e-03, 3.90944152e-03,
 9.05028839e-01, 2.52552166e-02, 1.15719157e-02, 9.73639295e-03,
 3.69359010e-08, 1.31913015e-08, 3.77811531e-08, 2.85698817e-08])



# App config.
DEBUG = True
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

class ReusableForm(Form):
    name = TextField('Name:', validators=[validators.DataRequired()])
    def adapt_array(arr):
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return sqlite3.Binary(out.read())

    def convert_array(text):
        out = io.BytesIO(text)
        out.seek(0)
        return np.load(out)

    sqlite3.register_adapter(np.ndarray, adapt_array)
    sqlite3.register_converter('array', convert_array)  
    @app.route("/", methods=['GET', 'POST'])
    @cross_origin()
    def hello():
        form = ReusableForm(request.form)
    
        print(form.errors)
        if request.method == 'POST':
            name=request.form['name']
            conn = sqlite3.connect('WDIdb.sqlite', detect_types=sqlite3.PARSE_DECLTYPES)
            cur = conn.cursor() 
            cur.execute('SELECT * FROM WDI WHERE name="' + name.lower() + '"')
            try:
                Data = cur.fetchone()
                df = pd.DataFrame(data=Data[4].T, index=Data[2].split(';'), columns=Data[3].split(';'))
                df = df.rename_axis('years').reset_index()
                data = []
                for i in range(36):
                    if np.mean(Data[4][i,:]) > 0:
                       trace = go.Bar(x=np.log(df[Data[3].split(';')[i]]), y=df['years'], orientation='h', name=Data[3].split(';')[i])
                       layout = go.Layout(yaxis=dict(title='year'), height=50000, width=1200, )
                       data.append(trace)
                fig = go.Figure(data=data, layout=layout)
                fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                #print(fig_json)            
                #model = keras.models.load_model('LeanIXML/')
                filename = 'WDI_model.sav'
                model = pickle.load(open(filename, 'rb'))
                #print (model)
                data = []
                for i in range((36)):
                    temp = Data[4][i,:]
                    temp = temp[temp != 0]
                    #print (temp.shape)
                    if (len(temp) > 0):
	                    regr.fit(temp.reshape(-1, 1), np.linspace(0,len(temp),len(temp)))
        	            data.append(regr.coef_)
                    else:
                    	data.append([0])
                data = np.asarray(data)
                #print (data.shape)
                data[:,0] = (data[:,0]-mean)/std
                pred = model.predict(data.reshape(1,-1))
                print(pred)    
                # Save the comment here.
                if pred == 1:
                    flash(name.capitalize() + ' is not Latin America and the Caribbean')
                else: 
                    flash(name.capitalize() + ' is Latin America and the Caribbean')

                return render_template('view.html', form=form, plot=fig_json)

            except (TypeError, AttributeError, ValueError, sqlite3.Error):

                print("An error occurred")
                if form.validate():
                     flash(name.capitalize() + ' is not in the database')
                else:
                     flash('...')
        return render_template('view.html', form=form)
if __name__ == "__main__":
    app.run()
