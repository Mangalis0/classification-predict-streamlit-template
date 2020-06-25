def data_preprocessor(df):
    '''
    For preprocessing we have cleaned the data, regularized it, transformed each cases, 
    tokenized the words from each setence/tweet and remove stopwords. For normalization, 
    we have used WordNetLemmatizer which transforms a sentence i.e. from this "love loving loved" 
    to this "love love love".
    
    '''
    stop_words = set(stopwords.words('english'))
    #stop_words.append(RT)
    #stemmer = PorterStemmer()
    lemm = WordNetLemmatizer()
    Tokenized_Doc=[]
    print("Preprocessing data.........\n")
    for data in df['message']:
        review = re.sub('[^a-zA-Z]', ' ', data)
        url = re.compile(r'https?://\S+|www\.\S+')
        review = url.sub(r'',review)
        html=re.compile(r'<.*?>')
        review = html.sub(r'',review)
        emojis = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
        review = emojis.sub(r'',review)
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(review)
        #gen_tweets = [stemmer.stem(token) for token in tokens if not token in stop_words]
        gen_tweets = [lemm.lemmatize(token) for token in tokens if not token in stop_words]
        cleaned =' '.join(gen_tweets)
        Tokenized_Doc.append(gen_tweets)
        df['tweet tokens'] = pd.Series(Tokenized_Doc)
        #df.insert(loc=2, column='tweet tokens', value=Tokenized_Doc)
    return df