# MACHINE-_LEARNING_MODEL_IMPLEMENTATION


*COMPANY* : CODETECH IT SOLUTIONS

*NAME* : MANDAPALLI GOPI

*INTERN ID* : CT04DN1772

*DURATION* : 4-WEEKS

*MENTOR* : NEELA SANTOSH

*DESCRIPTION FOR USING SCIKIT LEARN TO CLASSIFY FOR THE TASK* :


The given Python program is a simple implementation of a spam detection model using machine learning techniques in a Jupyter Notebook. The main goal of the code is to classify text messages into two categories: spam and ham (non-spam). The program begins by importing essential libraries including pandas for handling data, train_test_split from sklearn.model_selection for splitting the dataset, CountVectorizer from sklearn.feature_extraction.text for transforming text data into numerical features, and MultinomialNB from sklearn.naive_bayes to build the spam classifier. The accuracy_score and classification_report from sklearn.metrics are used to evaluate the model's performance.

Next, a small dataset is initialized manually using a Python dictionary containing sample text messages and their corresponding labels—either "spam" or "ham". This dataset is then converted into a pandas DataFrame for easier manipulation. To prepare the data for machine learning, the labels are encoded using a mapping function that converts "ham" to 0 and "spam" to 1. This binary classification setup helps the model distinguish between non-spam and spam messages during training and prediction.

The core of the feature extraction step is handled by CountVectorizer, which converts the text messages into a matrix of token counts. This step is crucial because machine learning algorithms cannot work directly with raw text—they require numerical input. CountVectorizer transforms each message into a numerical feature vector based on the frequency of each word in the message. This "bag-of-words" approach is commonly used in natural language processing (NLP) for tasks like text classification.

The dataset is then split into training and testing subsets using train_test_split. In this case, 70% of the data is used for training the model, while the remaining 30% is reserved for testing. The model used here is the MultinomialNB classifier, which is particularly well-suited for text classification tasks involving word counts. The model is trained on the training data using the fit() method, which learns the statistical relationships between the features (word counts) and the target labels (spam or ham).

After training, the model is tested using the unseen test data by calling the predict() method. The predicted values are then compared with the actual labels to evaluate how well the model performs. This evaluation is done using two metrics: accuracy score, which tells us the percentage of correctly classified messages, and the classification report, which provides detailed metrics such as precision, recall, and F1-score for both spam and ham classes.

Overall, this program demonstrates the fundamental steps of building a spam detection system using basic natural language processing and machine learning techniques. It shows how to preprocess text, convert it into numerical features, apply a machine learning algorithm, and evaluate the results. While the dataset is small and manually defined for simplicity, the structure of the program can be easily scaled up to work with larger and more complex datasets for real-world spam detection applications.


*OUTPUT PHOTOS* :

*PHOTO-1* :
![Image](https://github.com/user-attachments/assets/75ae6b77-9376-4a0a-be79-9337b9b4d540)

*PHOTO-2* :
![Image](https://github.com/user-attachments/assets/b958b859-13c2-445d-a100-0c2380426b73)

*PHOTO-3* :
![Image](https://github.com/user-attachments/assets/6345a7aa-b05b-49f3-ba8d-933c79e7066f)


*OUTPUT VIDEO* :
https://github.com/user-attachments/assets/6a679cdf-4871-43fc-9e4c-7fcbb375ad5b








