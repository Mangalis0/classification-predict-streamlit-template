#streamlit dependencies

import streamlit as st
import joblib, os

## data dependencies

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

##reading in the raw data and its cleaner

vectorizer = open('resources/tfidfvect.pkl','rb')   ##  will be replaced by the cleaning and preprocessing function
tweet_cv = joblib.load(vectorizer)

raw_data = raw_data = pd.read_csv('resources/train.csv')

def main():
    """Tweets classifier App"""

    st.title('Tweets  Unclassified  :)')
    st.subheader('Climate Change Belief Analysis: Based on Tweets')

    ##creating a sidebar for selection purposes


    pages = ['Information', 'Make Prediction', 'Contact App Developers']

    selection = st.sidebar.selectbox('Select Option', pages)

    ##information page

    if selection == 'Information':
        st.info('General Information')
        st.markdown('Explore Explorer Explorest.....boooom EXPLODE!!!!!!!!!!!')

    ## prediction page

    if selection == 'Make Prediction':

        st.info('Make Predictions of your Dataset with ML Model')

        data_source = ['Single text', 'Dataset'] ## differentiating between a single text and a dataset inpit

        source_selection = st.selectbox('Choose Option', data_source)

        if source_selection == 'Single text':
           # st.info('Classyfing a single text')

            text_input = st.text_area('Enter Text:', 'Type Here') ##user entering a single text to classify and predict
            
            if st.button('Classify'):

                vect_text = tweet_cv.transform([text_input]).toarray()##using tfidf to clean and preprocess

                predictor = joblib.load(open(os.path.join('resources/Logistic_regression.pkl'),'rb')) ##opening the stored model

                prediction = predictor.predict(vect_text)##making prediction

                prediction_dict = {'negative':-1,'Neutral':0, 'Positive':1}
                
                st.success('Result:  {}'.format(prediction))

                for i,j in prediction_dict.items():
                    if prediction == i:
                        st.success('Tweet has a {}'.format(j), 'Sentiment Towards Climate change')

    ##contact page
    if selection == 'Contact App Developers':

        st.info('Contact details if you any query:')
        st.write('kea: Lefifikea@gmail.com')
        st.write('Noxolo: ')
        st.write('Sam: ')
        st.write('Neli:')
        st.write('Ife: ')
        st.write('Khathu: ')

if __name__ == '__main__':
	main()


