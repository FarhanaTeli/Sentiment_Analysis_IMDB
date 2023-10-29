# Sentiment Classification for the IMDB Dataset using TF-IDF, USE, TF-IDF + USE, and Various Classifiers

The project aims to perform sentiment classification on the IMDB dataset by combining the TF-IDF (Term Frequency-Inverse Document Frequency) technique, Universal Sentence Encoder (USE), TF-IDF + USE and different classifiers such as LinearSVM, Logistic Regression, Naive Bayes, XGBoost and Random Forest. The project follows a multi-step process:

**Step 1:** Preprocessing and Data Loading

--> Load the IMDB dataset from a CSV file.

--> Preprocess the text data by removing HTML tags, stripping whitespace, eliminating noisy characters, reducing extra spaces, and converting text to lowercase.

--> Tokenize the text and remove stopwords using NLTK.

--> Split the dataset into training and testing sets.

**Step 2:** TF-IDF Vectorization

--> Use the TF-IDF vectorization technique to convert the preprocessed text data into numerical features.

--> Limit the number of features to 5000 using max_features.

--> Transform both the training and testing sets using the TF-IDF vectorizer.

**Step 3:** USE (Universal Sentence Encoder)

--> Load the Universal Sentence Encoder (USE) from TensorFlow Hub.

--> Embed the text data using USE, generating dense vector representations of sentences.

--> Create USE embeddings for both the training and testing sets.

**Step 4:** TF-IDF + USE (Universal Sentence Encoder)
 
--> Combined both embeddings for both training and testing sets.

**Step 5:** Classification with Various Classifiers

--> Initialize various classifiers, including LinearSVM, Logistic regression, Naive Bayes, XGBoost, and Random Forest.

--> Train each classifier on the TF-IDF-transformed training data.

--> Make predictions using each classifier on the TF-IDF-transformed testing data.

--> Calculate and display the accuracy, precision, recall, F1 score, classification report of each classifier's predictions. The same approach used for other two embeddings such as USE and TF-IDF + USE.

The project provides a comprehensive analysis of sentiment classification on the IMDB dataset by comparing the performance of different classifiers using TF-IDF, USE, and TF-IDF + USE representations. This allows for the selection of the most suitable approach for sentiment analysis.

The code ensures that the data is preprocessed, vectorized, and classified effectively, providing insights into the best method for sentiment classification on the IMDB dataset. Additionally, the use of PyTorch and TensorFlow for different parts of the project demonstrates the flexibility of working with deep learning frameworks.
