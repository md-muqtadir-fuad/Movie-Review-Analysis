# Movie Review Analysis
🎬 **Movie Review Sentiment Analysis**

This Python script performs sentiment analysis on movie reviews using the IMDB Dataset. The dataset is loaded from a CSV file using Pandas and preprocessed to prepare the text data for analysis. The preprocessing steps include tokenization, removal of non-alphabetic characters, lowercase conversion, and stemming.

The sentiment analysis model is based on a Multinomial Naive Bayes classifier and utilizes the Bag of Words (BoW) approach. The CountVectorizer is used to convert the text data into a matrix of token counts, and a LabelBinarizer is employed to transform sentiment labels into binary form.

The model is trained on a subset of the dataset and evaluated on the remaining data. The accuracy score for the sentiment analysis is calculated using the scikit-learn library.

**Key Steps:**
1. **Data Loading and Exploration:**
   - Load the IMDB Dataset from the provided URL.
   - Display the first 5 rows and basic statistics of the dataset.

2. **Text Preprocessing:**
   - Tokenize the reviews using NLTK.
   - Remove non-alphabetic characters and convert text to lowercase.
   - Remove stopwords and apply stemming to the words.

3. **Train-Test Split:**
   - Split the dataset into training and testing sets.

4. **Vectorization:**
   - Use CountVectorizer to transform the reviews into a bag of words representation.
   - LabelBinarizer is applied to transform sentiment labels into binary form.

5. **Model Training:**
   - Train a Multinomial Naive Bayes classifier on the bag of words representation.

6. **Evaluation:**
   - Predict sentiment on the test set.
   - Calculate and print the accuracy score.

Feel free to use or contribute to this project for further enhancements in movie review sentiment analysis!


[https://github.com/md-muqtadir-fuad/Movie-Review-Analysis]

