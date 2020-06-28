#streamlit dependencies

import streamlit as st
import joblib, os

## data dependencies

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

import seaborn as sns
import re


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

    selection = st.sidebar.radio('Go to....', pages)

    #st.sidebar.image(image, caption='Which Tweet are you?', use_column_width=True)



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

    
        

       # Number of Messages Per Sentiment
        st.write('Distribution of the sentiments')
        # Labeling the target
        data['sentiment'] = [['Negative', 'Neutral', 'Positive', 'News'][x+1] for x in data['sentiment']]
        
        # checking the distribution
        st.write('The numerical proportion of the sentiments')
        values = data['sentiment'].value_counts()/data.shape[0]
        labels = (data['sentiment'].value_counts()/data.shape[0]).index
        plt.pie(x=values, labels=labels, autopct='%1.1f%%', startangle=90, explode= (0.04, 0, 0, 0))
        st.pyplot()
        
        # checking the distribution
        sns.countplot(x='sentiment' ,data = data, palette='PRGn')
        plt.ylabel('Count')
        plt.xlabel('Sentiment')
        plt.title('Number of Messages Per Sentiment')
        st.pyplot()

        # Popular Tags
        st.write('Popular tags found in the tweets')
        data['users'] = [''.join(re.findall(r'@\w{,}', line)) if '@' in line else np.nan for line in data.message]
        sns.countplot(y="users", hue="sentiment", data=data,
                    order=data.users.value_counts().iloc[:20].index, palette='PRGn') 
        plt.ylabel('User')
        plt.xlabel('Number of Tags')
        plt.title('Top 20 Most Popular Tags')
        st.pyplot()

        # Tweet lengths
        st.write('The length of the sentiments')
        sns.barplot(x=['sentiment'], y=['message'].apply(len), data = data, palette='PRGn')
        plt.ylabel('Length')
        plt.xlabel('Sentiment')
        plt.title('Average Length of Message by Sentiment')
        st.pyplot()

        # Generating the word cloud image from all the messages
        from wordcloud import WordCloud, ImageColorGenerator
        text = " ".join(tweet for tweet in data.message)   # Combining all the messages
        wordcloud = WordCloud(background_color="white").generate(text)
        # Displaying the word cloud image using matplotlib:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot()

    ## prediction page

    if selection == 'Make Prediction':

        st.info('Make Predictions of your Tweet(s) using our ML Model')

        data_source = ['Select option', 'Single text', 'Dataset'] ## differentiating between a single text and a dataset inpit

        source_selection = st.selectbox('What to classify?', data_source)

        # Load Our Models
        def load_prediction_models(model_file):
            loaded_models = joblib.load(open(os.path.join(model_file),"rb"))
            return loaded_models

        # Getting the predictions
        def get_keys(val,my_dict):
            for key,value in my_dict.items():
                if val == value:
                    return key


        if source_selection == 'Single text':
            ### SINGLE TWEET CLASSIFICATION ###
            st.subheader('Single tweet classification')

            input_text = st.text_area('Enter Text (max. 140 characters):') ##user entering a single text to classify and predict
            all_ml_models = ["LR","NB","RFOREST","DECISION_TREE"]
            model_choice = st.selectbox("Choose ML Model",all_ml_models)

            st.info('for more information on the above ML Models please visit: https://datakeen.co/en/8-machine-learning-algorithms-explained-in-human-language/')

            prediction_labels = {'Negative':-1,'Neutral':0,'Positive':1,'News':2}
            if st.button('Classify'):

                st.text("Original test ::\n{}".format(input_text))
                vect_text = tweet_cv.transform([input_text]).toarray()
                if model_choice == 'LR':
                    predictor = load_prediction_models("resources/Logistic_regression.pkl")
                    prediction = predictor.predict(vect_text)
                    # st.write(prediction)
                elif model_choice == 'RFOREST':
                    predictor = load_prediction_models("resources/RFOREST_model.pkl")
                    prediction = predictor.predict(vect_text)
                    # st.write(prediction)
                elif model_choice == 'NB':
                    predictor = load_prediction_models("resources/NB_model.pkl")
                    prediction = predictor.predict(vect_text)
                    # st.write(prediction)
                elif model_choice == 'DECISION_TREE':
                    predictor = load_prediction_models("resources/DTrees_model.pkl")
                    prediction = predictor.predict(vect_text)
				# st.write(prediction)

                final_result = get_keys(prediction,prediction_labels)
                st.success("Tweet Categorized as:: {}".format(final_result))

        if source_selection == 'Dataset':
            ### DATASET CLASSIFICATION ###
            st.subheader('Dataset tweet classification')

            all_ml_models = ["LR","NB","RFOREST","DECISION_TREE"]
            model_choice = st.selectbox("Choose ML Model",all_ml_models)

            st.info('for more information on the above ML Models please visit: https://datakeen.co/en/8-machine-learning-algorithms-explained-in-human-language/')


            prediction_labels = {'Negative':-1,'Neutral':0,'Positive':1,'News':2}
            text_input = st.file_uploader("Choose a CSV file", type="csv")
            if text_input is not None:
                text_input = pd.read_csv(text_input)

            uploaded_dataset = st.checkbox('See uploaded dataset')
            if uploaded_dataset:
                st.dataframe(text_input.head(25))

            col = st.text_area('Enter column to classify')

            #col_class = text_input[col]
            
            if st.button('Classify'):

                st.text("Original test ::\n{}".format(text_input))
                vect_text = tweet_cv.transform([text_input[col]]).toarray()
                if model_choice == 'LR':
                    predictor = load_prediction_models("resources/Logistic_regression.pkl")
                    prediction = predictor.predict(vect_text)
                    # st.write(prediction)
                elif model_choice == 'RFOREST':
                    predictor = load_prediction_models("resources/RFOREST_model.pkl")
                    prediction = predictor.predict(vect_text)
                    # st.write(prediction)
                elif model_choice == 'NB':
                    predictor = load_prediction_models("resources/NB_model.pkl")
                    prediction = predictor.predict(vect_text)
                    # st.write(prediction)
                elif model_choice == 'DECISION_TREE':
                    predictor = load_prediction_models("resources/DTrees_model.pkl")
                    prediction = predictor.predict(vect_text)


                
				# st.write(prediction)
                text_input['sentiment'] = prediction
                final_result = get_keys(prediction,prediction_labels)
                st.success("Tweets Categorized as:: {}".format(final_result))

                
                csv = text_input.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
                href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'

                st.markdown(href, unsafe_allow_html=True)


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


