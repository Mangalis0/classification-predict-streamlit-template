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

#@st.cache
data = pd.read_csv('resources/train.csv')

def main():
    """Tweets classifier App"""

    st.title('Tweets  Unclassified  :)')

    from PIL import Image
    image = Image.open('resources/imgs/Tweeter.png')

    st.image(image, caption='Which Tweet are you?', use_column_width=True)

    st.subheader('Climate Change Belief Analysis: Based on Tweets')
    

    ##creating a sidebar for selection purposes


    pages = ['Information', 'Visuals', 'Make Prediction', 'Contact App Developers']

    selection = st.sidebar.selectbox('Select Option', pages)

    ##information page

    if selection == 'Information':
        st.info('General Information')
        st.write('Explorers Explore and.....boooom EXPLODE!!!!!!!!!!!')
        st.markdown(""" We have deployed Machine Learning models that are able to classify 
        whether or not a person believes in climate change, based on their novel tweet data. 
        Like any data lovers, these are robust solutions to that can provide access to a 
        broad base of consumer sentiment, spanning multiple demographic and geographic categories. 
        So, do you have a Twitter API and ready to scrap? or just have some tweets off the top of your head? 
        Do explore the rest of this app's buttons.
        """)


        raw = st.checkbox('See raw data')
        if raw:
            st.dataframe(data.head(25))

    ## Charts page

    if selection == 'Visuals':
        st.info('The following are some of the charts that we have created from the raw data')
        #st.markdown(''' The following are some of the charts that we have created from the raw data''')

        import plotly.express as px
        f = px.histogram(data['sentiment'])
        f.update_xaxes(title="sentiment")
        f.update_yaxes(title="No. of tweets")
        st.plotly_chart(f)

        st.write("where -1 represents Negative views ")
        st.write("where 0 represents Neutral views ")
        st.write("where 1 represents Positive views ")
        st.write(" and where 2 represents News outlets ")

    ## prediction page

    if selection == 'Make Prediction':

        st.info('Make Predictions of your Tweet(s) using our ML Model')

        data_source = ['Select option', 'Single text', 'Dataset'] ## differentiating between a single text and a dataset inpit

        source_selection = st.selectbox('What to classify?', data_source)

        if source_selection == 'Single text':
            ### SINGLE TWEET CLASSIFICATION ###
            st.subheader('Single tweet classification')

            text_input = st.text_area('Enter Text:') ##user entering a single text to classify and predict

            if st.button('Classify'):

                vect_text = tweet_cv.transform([text_input]).toarray()##using tfidf to clean and preprocess

                predictor = joblib.load(open(os.path.join('resources/Logistic_regression.pkl'),'rb')) ##opening the stored model

                prediction = predictor.predict(vect_text)##making prediction

                prediction_dict = {'negative':-1,'Neutral':0, 'Positive':1}
                
                st.success('Result:  {}'.format(prediction))

        if source_selection == 'Dataset':
            ### DATASET CLASSIFICATION ###
            st.subheader('Dataset tweet classification')

            text_input = st.file_uploader("Choose a CSV file", type="csv")
            if text_input is not None:
                text_input = pd.read_csv(text_input)
            #st.write(df)
        
            
            if st.button('Classify'):

                #classifier = st.selectbox('Which algorithm?', alg)
                #if classifier=='Decision Tree':

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

        st.info('Contact details in case you any query or would like to know more of our designs:')
        st.write('Kea: Lefifikea@gmail.com')
        st.write('Noxolo: Kheswanl925@gmail.com')
        st.write('Sam: makhoba808@gmail.com')
        st.write('Neli: cenygal@gmail.com')
        st.write('Khathu: netsiandakhathutshelo2@gmail.com')
        st.write('Ife: ifeadeoni@gmail.com')

if __name__ == '__main__':
	main()


