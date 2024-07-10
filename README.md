# CODTECH-Task2

### **Name**: Harsha Vardhini V
### **Company**: CODTECH IT SOLUTIONS
### **ID**: CT08DS1883
### **Domain**: MACHINE LEARNING
### **Duration**: JUNE to JULY 2024
### **Mentor**: SRAVANI GOUNI

### Overview of the Project

**Project**: ANALYSIS ON MOVIE REVIEWS

**Objective**: The objective of this project is to build a machine learning model that can accurately predict the sentiment (positive or negative) of movie reviews. This model aims to help in understanding the general perception of movies based on textual reviews, which can be beneficial for movie producers, critics, and viewers.

**Key Activities**:
1. **Data Loading and Inspection**:
   - Load the IMDB movie reviews dataset and inspect its structure.
   - Check for data types, missing values, and basic statistics.

2. **Data Pre-processing**:
   - **Text Normalization**: Convert text to lowercase to ensure uniformity.
   - **Removing Noisy Text**: Remove HTML tags and special characters to clean the text.
   - **Text Stemming**: Apply stemming to reduce words to their base form.
   - **Stop Words Removal**: Remove common stop words that do not contribute to sentiment.
   - **Word Embedding**: Use techniques like Count Vectorizer and TF-IDF to convert text into numerical features.

3. **Data Preparation**:
   - Split the dataset into training and testing sets to evaluate the model's performance.

4. **Model Building**:
   - Use Support Vector Machine (SVM) to build the sentiment analysis model.
   - Train the model on the training dataset.

5. **Model Evaluation**:
   - Evaluate the model's performance on the test dataset using metrics like accuracy, precision, recall, and F1-score.
   - Analyze the confusion matrix to understand the model's prediction errors.

**Technologies Used**:
- **Programming Languages**: Python
- **Libraries**:
  - Data Manipulation: Pandas, Numpy
  - Text Pre-processing: re, NLTK
  - Machine Learning: Scikit-learn
  - Model Evaluation: Scikit-learn metrics

**Key Insights**:
1. **Data Quality**: The dataset was clean with no missing values, but the text required significant pre-processing to remove noise and standardize the format.
2. **Feature Extraction**: Using techniques like Count Vectorizer and TF-IDF helped in transforming the textual data into numerical features, which are suitable for machine learning models.
3. **Model Performance**: The SVM model provided a good balance between precision and recall, indicating its effectiveness in predicting sentiment accurately.
4. **Error Analysis**: The confusion matrix and classification report provided insights into the model's strengths and weaknesses, highlighting areas for potential improvement.
5. **Practical Application**: The sentiment analysis model can be used by movie producers to gauge audience reactions, by critics to analyze trends, and by viewers to get a sense of the general sentiment towards a movie before watching it.

