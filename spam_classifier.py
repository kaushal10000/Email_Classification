import numpy as np
import pandas as pd
import string
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import seaborn as sns
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

df = pd.read_csv(r'C:\Users\kaush\Spam-Classification\spam.csv', encoding='ISO-8859-1')
print("Random Data from complete Dataset: ")
print(df.sample(5))
print("Size of Dataset: ", df.shape)
print("\n")

df.info()
print("Dropping Unnamed Columns from the Dataset ")
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
print(df.sample(5))
print("\n")

print("Renamming Columns v1 and v2 to target and text.")
df.rename(columns={'v1':'target','v2':'text'},inplace=True)
print(df.sample(5))
print("\n")

df['target'] = encoder.fit_transform(df['target'])
print(df.head())
print("\n")
print("Missing Values in Dataset:")
p1=df.isnull().sum()
print(p1)
p2=df.duplicated().sum()
print("Duplicate Values in  the Dataset:",p2)

df = df.drop_duplicates(keep='first')
p3=df.duplicated().sum()
print("Duplicate Values after Dropping all Duplicate Values:",p3)
print("Final Size of Dataset: ",df.shape)
print("\n")

print(df.head())
print("Total Ham/Spam(0/1) ",df['target'].value_counts())
plt.pie(df['target'].value_counts(), labels=['ham','spam'],autopct="%0.2f")
plt.title('Distribution of Ham and Spam Emails')
plt.show()

nltk.download('punkt')
print("Dispay Number of Character: ")
df['num_characters'] = df['text'].apply(len)
print(df.head())
print("\n")
print("Dispay Number of Words: ")
df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))
print(df.head())
print("\n")
print("Dispay Number of Sentences: ")
df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))
print(df.head())
print("\n")

s=df[['num_characters','num_words','num_sentences']].describe()
print("Statistics of Number of Character, Words and Sentences: ")
print(s)
print("\n")

s1=df[df['target'] == 0][['num_characters','num_words','num_sentences']].describe()
s2=df[df['target'] == 1][['num_characters','num_words','num_sentences']].describe()
print("Ham Statistics: ")
print(s1)
print("Spam Statistics: ")
print(s2)
print("\n")

plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_characters'])
sns.histplot(df[df['target'] == 1]['num_characters'],color='red')
plt.title('Distribution of Number of Characters in Ham Vs Spam Emails')
plt.xlabel('Number of Characters')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_words'])
sns.histplot(df[df['target'] == 1]['num_words'],color='red')
plt.title('Distribution of Number of Words in Ham Vs Spam Emails')
plt.xlabel('Number of Characters')
plt.ylabel('Frequency')
plt.show()

sns.pairplot(df,hue='target')
plt.title('Pairplot of Features Colored by Spam/Ham', y=1.02)
plt.show()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)

df['transformed_text'] = df['text'].apply(transform_text)
print("Dataset after change it into lowercase, removing special characters, stopwords, punctaion and doing stemming")
print(df.head())
print("\n")

wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep=" "))
plt.figure(figsize=(15,6))
plt.imshow(spam_wc, interpolation='bilinear')
plt.axis('off')  
plt.title('Word Cloud for Spam Emails')
plt.show()  

ham_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep=" "))
plt.figure(figsize=(15,6))
plt.imshow(ham_wc, interpolation='bilinear')
plt.axis('off')  
plt.title('Word Cloud for Normal Emails')
plt.show()  

print(df.head)
print("\n")
spam_corpus = []
for msg in df[df['target'] == 1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)

print("No. of Words in Spam Mails are ",len(spam_corpus))

word_counts = Counter(spam_corpus).most_common(30)
df_word_counts = pd.DataFrame(word_counts, columns=['Word', 'Count'])
sns.barplot(x='Word', y='Count', data=df_word_counts)
plt.title('Top 30 Most Common Words in Spam Mails')
plt.xticks(rotation=90)
plt.show()

ham_corpus = []
for msg in df[df['target'] == 0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)

print("No. of Words in Normal Mails are ",len(ham_corpus))

word_counts = Counter(ham_corpus).most_common(30)

df_word_counts = pd.DataFrame(word_counts, columns=['Word', 'Count'])
sns.barplot(x='Word', y='Count', data=df_word_counts)
plt.title('Top 30 Most Common Words in Normal Mails')
plt.xticks(rotation=90)
plt.show()

print(df.head())
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['transformed_text']).toarray()
print(X.shape)
y = df['target'].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)
print("\n")

print("Gaussian Naive Bayes: ")
gnb.fit(X_train,y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))
print("\n")

print("Multinomial Naive Bayes: ")
mnb.fit(X_train,y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))
print("\n")


print("Gaussian Naive Bayes: ")
bnb.fit(X_train,y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))
print("\n")

svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)
xgb = XGBClassifier(n_estimators=50,random_state=2)

clfs = {
    'SVC' : svc,
    'KN' : knc, 
    'NB': mnb, 
    'DT': dtc, 
    'LR': lrc, 
    'RF': rfc, 
    'AdaBoost': abc, 
    'BgC': bc, 
    'ETC': etc,
    'GBDT':gbdt,
    'xgb':xgb
}

def train_classifier(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    return accuracy, precision

accuracy_scores = []
precision_scores = []

for name, clf in clfs.items():
    current_accuracy, current_precision = train_classifier(clf, X_train, y_train, X_test, y_test)
    
    print("For ", name)
    print("Accuracy - ", current_accuracy)
    print("Precision - ", current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)

performance_df = pd.DataFrame({
    'Algorithm': list(clfs.keys()),  
    'Accuracy': accuracy_scores,
    'Precision': precision_scores
})

performance_df = performance_df.sort_values('Precision', ascending=False)

print("\nPerformance of classifiers:")
print(performance_df)