#!/usr/bin/env python
# coding: utf-8

# In[1]:




# In[1]:


from flask import Flask,render_template,request
import numpy as np
import joblib


# In[2]:


#import ML Model
model=joblib.load("iris_model.pkl")


# In[3]:


#flask initializaion 
app=Flask(__name__)


# In[4]:


#route for input of web 
@app.route('/')
def home():
    return render_template('index.html')


# In[5]:


#prediction 
@app.route('/predict',methods=['POST'])
def predict():
    sepal_length=float(request.form['sepal_length'])
    sepal_width=float(request.form['sepal_width'])
    petal_length=float(request.form['petal_length'])
    petal_width=float(request.form['petal_width'])

    features=np.array([[sepal_length,sepal_width,petal_length,petal_width]])
    prediction_label=model.predict(features)[0]
    species=['Setosa','Versicolor','Virginica'][prediction_label]
    return render_template('result.html',prediction=species)
    


# In[ ]:


#run app
if(__name__)=="__main__":
    app.run(debug=True, use_reloader=False)


# In[ ]:




