import streamlit as st
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer


# Load the pre-trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Open the vectorizer file
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

tab1, tab2, tab3 = st.tabs(["Classify", "Data analysis", "Accuracy of the model"])

# Define the Streamlit app
def main():
    with tab3:
        st.title("Accuracy of the model")
        st.text("The model has an accuracy of 95.96%")
        st.image("confusion_matrix.png")

    with tab2:
        st.title("Fake News Data Vistualization")  
        st.image("fake_genuine_distribution.png", caption="Fake vs Genuine")
        st.image("fake_used_words.png", caption="Fake news most used words")
        st.image("genuine_used_words.png", caption="Genuine news most used words")

    with tab1:
        st.title("Fake News Classifier")
        st.subheader("by Ismael Julian Vila")

        # Get user input
        article = st.text_area("Enter the news article", "")

        # Classify the article
        if st.button("Classify"):
            if article:
                prediction = predict_fake_news(article)   
                st.write(f"The article is {prediction}")
            else:
                st.write("Please enter a news article.")

def predict_fake_news(article):
    # Clean the user input
    cleaned_user_text = []

    # Load the lemmatizer and stopwords
    lemma = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    words = nltk.word_tokenize(article)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    words = [lemma.lemmatize(w) for w in words]
    cleaned_user_text.append(' '.join(words))

    # Transform the cleaned user input using the CountVectorizer
    user_input_bow = vectorizer.transform(cleaned_user_text)

    # Make the prediction
    prediction = model.predict(user_input_bow)

    # Print the prediction
    if prediction[0] == 1:
        return "Fake"
    else:
        return "Genuine"

# Run the app
if __name__ == "__main__":
    main()
