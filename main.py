import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write('''
# Application Simple pour le Prévision des fleurs d'Iris
cette application prédit la catégorie des fleurs d'Iris 
''')

st.sidebar.header("Les parametres d'entrée")

def user_input():
    sepal_length=st.sidebar.slider('La longeur du Sepal ',4.3,7.9)
    sepal_width=st.sidebar.slider('La largeur du Sepal ',2.0,4.4,3.3)
    petal_length=st.sidebar.slider('La longeur du petal ',1.0 ,6.9,2.3)
    petal_width=st.sidebar.slider('La largeur du petal ',0.1,2.5,1.3)

    ##On definit un dictionnaires

    data={'sepal_length':sepal_length,
          'sepal_width':sepal_width,
          'petal_length':petal_length,
          'petal_width':petal_width }

    fleurs_parametres=pd.DataFrame(data,index=[0])
    return fleurs_parametres


df=user_input()

st.subheader('on veut trouver la catégoeire de cette fleur')
st.write(df)

iris=datasets.load_iris()

#On créer notre Objet Model
clf=RandomForestClassifier()
clf.fit(iris.data,iris.target)
#le syntaxe Génerale de La commande fit
#model.fit(X_train, y_train)
#pour entraîner un modèle sur un ensemble de données.


prediction=clf.predict(df)
st.subheader("La catégorie de la fleuur d'iris est :")
st.write(iris.target_names[prediction])


