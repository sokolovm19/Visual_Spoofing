import sys
import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# usage: ./experiment.py data_file experiment
# e.g.,  ./experiment.py spam_ham_dataset.csv A
#        ./experiment.py spam_ham_dataset.csv B
#        ./experiment.py spam_ham_dataset.csv C

def find_features(message):
    words = word_tokenize(message)
    features = {}
    for word in word_features:
        features[word] = (word in words)
    return features

# read the data from file
df = pd.read_csv(sys.argv[1], header=None, encoding='utf-8')
#df = pd.read_csv('spam_ham_dataset.csv', header=None, encoding='utf-8')

# replace email addresses, URLs, £, $, phone numbers, and numbers with placeholders
df[1] = df[1].str.replace(r'^.+@[^\.].*\.[a-z]{2,}$', 'emailaddress')
df[1] = df[1].str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$', 'webaddress')
df[1] = df[1].str.replace(r'£|\$', 'moneysymb')
df[1] = df[1].str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$', 'phonenumbr')
df[1] = df[1].str.replace(r'\d+(\.\d+)?', 'numbr')
# replace anything that doesn't start with letters, digits, or spaces with a space
df[1] = df[1].str.replace(r'[^\w\d\s]', ' ')
# replace multiple spaces with one space
df[1] = df[1].str.replace(r'\s+', ' ')
# remove empty lines
df[1] = df[1].str.replace(r'^\s+|\s+?$', '')

# remove stop words
stop_words = set(stopwords.words('english'))
df[1] = df[1].apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))

# stem the words
ps = nltk.PorterStemmer()
df[1] = df[1].apply(lambda x: ' '.join(ps.stem(term) for term in x.split()))

# replace listed characters based on experiment: A) none, B) in the testing set, or C) in both sets
if (sys.argv[2] == "C"):
    print('Experiment C: replace characters in both set.')
    df[1] = df[1].str.replace('a', '\u0430')
    df[1] = df[1].str.replace('e', '\u0435')
    df[1] = df[1].str.replace('k', '\u043A')
    df[1] = df[1].str.replace('o', '\u043E')
    df[1] = df[1].str.replace('p', '\u0440')
    df[1] = df[1].str.replace('c', '\u0441')
    df[1] = df[1].str.replace('y', '\u0443')
elif(sys.argv[2] == "A"):
    print("Experiment A: don't replace any characters.")

# split the data into training and test sets
training, testing = model_selection.train_test_split(df, test_size=0.2, random_state=1)
if (sys.argv[2] == "B"):
    print('Experiment B: replace characters in testing set.')
    for key in pd.DataFrame.keys(testing[1]):
        testing[1][key] = testing[1][key].replace('a', '\u0430')
        testing[1][key] = testing[1][key].replace('e', '\u0435')
        testing[1][key] = testing[1][key].replace('k', '\u043A')
        testing[1][key] = testing[1][key].replace('o', '\u043E')
        testing[1][key] = testing[1][key].replace('p', '\u0440')
        testing[1][key] = testing[1][key].replace('c', '\u0441')
        testing[1][key] = testing[1][key].replace('y', '\u0443')

# generate the dictionary for this corpus
all_words = []
for message in testing[1]:
    words = word_tokenize(message)
    for w in words:
        all_words.append(w)
for message in training[1]:
    words = word_tokenize(message)
    for w in words:
        all_words.append(w)

print("Number of features: {}".format(len(all_words)))

# create the features from this dictionary
all_words = nltk.FreqDist(all_words)
word_features = list(nltk.FreqDist(all_words).keys())

# generate features from the words in emails
encoder = LabelEncoder()
    # testing emails
Y = encoder.fit_transform(testing[0]) # the labels are in the first column
email = list(zip(testing[1], Y))
testing_fs = [(find_features(text), label) for (text, label) in email]
    # training emails
Y = encoder.fit_transform(training[0]) # the labels are in the first column
email = list(zip(training[1], Y))
training_fs = [(find_features(text), label) for (text, label) in email]

# create a "dictionary of classifiers"
names = ["Decision Tree", "Random Forest", "Naive Bayes","SVM Linear"]
classifiers = [
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    MultinomialNB(),
    SVC(kernel = 'linear')
]
models = zip(names, classifiers)

# train and evaluate these classifiers
for name, model in models:
    print(name)
    nltk_model = SklearnClassifier(model)
    nltk_model.train(training_fs)
    # print accuracy
    accuracy = nltk.classify.accuracy(nltk_model, testing_fs)*100
    print("Accuracy: {}".format(accuracy))
    # print classification report
    instances, labels = zip(*testing_fs)
    prediction = nltk_model.classify_many(instances)
    print('Classification report')
    print(classification_report(labels, prediction))
    # print confusion matrix
    print('Confusion matrix')
    print(pd.DataFrame(confusion_matrix(labels, prediction), index = [['actual', 'actual'], ['ham', 'spam']], columns = [['predicted', 'predicted'], ['ham', 'spam']]))
