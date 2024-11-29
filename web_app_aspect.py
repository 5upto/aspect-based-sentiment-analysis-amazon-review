import pickle
import re
import nltk
import streamlit as st
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

predictor = pickle.load(open("C:/Users/SAI RAM GANDHI/ML Project/model_dt.pkl", "rb"))
scaler = pickle.load(open("C:/Users/SAI RAM GANDHI/ML Project/scaler.pkl", "rb"))
cv = pickle.load(open("C:/Users/SAI RAM GANDHI/ML Project/countVectorizer.pkl", "rb"))
aspects = ["quality", "price","performance","design"]

def split_sentence_on_aspects(sentence, aspects):
    # Create a regex pattern to match aspect words as separate words (case insensitive)
    pattern = r'\b(' + '|'.join(aspects) + r')\b'
    
    # Find all occurrences of aspect words and their positions
    matches = [(match.group(), match.start()) for match in re.finditer(pattern, sentence, re.IGNORECASE)]
    
    # If no aspect words found, return the whole sentence as a single entry
    if not matches:
        return [["general", sentence]]
    
    # Store result in list-of-lists format with aspect and corresponding sentence part
    result = []
    start = 0
    for i, (word, pos) in enumerate(matches):
        # Find the end of the current aspect's sentence part
        end = matches[i + 1][1] if i + 1 < len(matches) else len(sentence)
        
        # Capture the part from the current aspect position to the next
        part = sentence[pos:end].strip()
        result.append([word, part])
        
        # Update start for the next loop iteration
        start = end
    
    return result

def prediction(predictor, scaler, cv, text_input):        
    corpus = []
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
    review = " ".join(review)
    corpus.append(review)
    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict(X_prediction_scl)
    return "Positive" if y_predictions == 1 else "Negative"



def main():    
    #Title
    st.title('Aspect Based Sentiment Analysis Web App') 
    
    #Getting the input data from user
    text_input = st.text_input("Enter your review here :")    
    
    #Code for Prediction
    sentiment = ''
    sentence_aspect = ''
    overall_sentiment = ''
    #Creating a button for prediction
    result = split_sentence_on_aspects(text_input, aspects)
    if st.button('Predict Sentiment'):
        st.write('Overall Sentiment :')
        overall_sentiment = prediction(predictor, scaler, cv, text_input)
        st.success(overall_sentiment)
        for aspect, sentence_part in result:
            sentence_aspect = aspect 
            sentiment = prediction(predictor, scaler, cv, sentence_part)
            st.write("Aspect : ")
            st.success(sentence_aspect)
            st.write("Sentiment : ")
            st.success(sentiment)
    
if __name__ == '__main__':
    main()
    
        
    
