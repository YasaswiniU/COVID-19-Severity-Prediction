from flask import Flask,render_template,url_for,request
from werkzeug.utils import secure_filename
import os,shortuuid,pickle
import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing import image
from keras.models import load_model
from datetime import datetime

sns.set()

app = Flask(__name__)
  
app.config['DATA_DIR'] = os.path.realpath(os.path.dirname(__file__))
AB_Model = pickle.load(open('models/adaboost.sav', 'rb'))
CNN_Model = load_model("models/severity_model_cnn.h5")


@app.route('/')
def home():
   return render_template('Main.html')

@app.route('/blood_samples',methods=['GET','POST'])
def blood_samples():
   if(request.method=='POST'):
      data=dict()
      name = request.form['name']
      data['Age'] = request.form['age']
      data['BUN'] = request.form['BUN']
      data['CrctProtein'] = request.form['CrctProtein']
      data['Creatinine'] = request.form['Creatinine']
      data['Ddimer'] = request.form['Ddimer']      
      result = AB_Model.predict(pd.DataFrame(data,index=[0]))
      res = dict(enumerate(result.flatten(), 1))[1]
      return render_template('result.html',data=data,result=res,name=name,date = str(datetime.now()),blood=True)
   return render_template('blood_samples.html')

@app.route('/chest-xray',methods=['GET','POST'])
def chest_xray():
   if(request.method=='POST'):
      f = request.files['img']
      path = os.path.join('static/FILE_UPLOADS/')
      path = path.replace('\\','/')
      filename = (f.filename)
      file_path = os.path.join(path,secure_filename(f.filename))
      if(os.path.exists(file_path)):
         filename  = shortuuid.uuid()+extract_extension(f.filename)
         f.save(os.path.join(path,secure_filename(filename)))
      else:
         f.save(file_path)

      test_image = image.load_img(os.path.join(path,filename), color_mode="grayscale", target_size = (256,256))
      test_image = image.img_to_array(test_image)
      test_image = np.expand_dims(test_image, axis = 0)
      result = CNN_Model.predict(test_image)
      if result[0][0] == 1:
        prediction = 'NON-CRITICAL'
      else:
        prediction = 'CRITICAL'

      name = request.form['name']

      return render_template('result.html',data=os.path.join(path,filename),result=prediction,name=name,date=str(datetime.now()),blood=False)

   return render_template('Xray.html')


def extract_extension(filename):
   file,extension = os.path.splitext(filename)
   return extension