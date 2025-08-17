**EMAIL SPAM CLASSIFICATION**

_This project leverages a variety of machine learning algorithms to classify emails as spam or not spam, each contributing a unique approach to solving the problem._

_1.	SVC and Logistic Regression utilize mathematical models to identify email patterns and make classifications based on these features._

_2.	K-Neighbors Classifier predicts spam by evaluating the similarity to nearby emails._

_3.	Decision Trees, along with ensemble methods like Random Forest and Gradient Boosting, combine multiple models to enhance prediction accuracy and reduce errors._

_4.	Naive Bayes and XGBoost focus on probabilistic models, optimizing performance for large datasets._

_5.	AdaBoost and Bagging are employed to adjust for difficult cases and reduce the risk of overfitting._

**Classifier Performance Summary**

_This analysis compares the performance of various machine learning classification algorithms using Accuracy and Precision as evaluation metrics._

**Results**

| Algorithm | Accuracy  | Precision |
|-----------|-----------|-----------|
| KN        | 0.905222  | 1.000000  |
| NB        | 0.970986  | 1.000000  |
| RF        | 0.975822  | 0.982906  |
| SVC       | 0.975822  | 0.974790  |
| ETC       | 0.974855  | 0.974576  |
| LR        | 0.958414  | 0.970297  |
| AdaBoost  | 0.960348  | 0.929204  |
| xgb       | 0.967118  | 0.926230  |
| GBDT      | 0.946809  | 0.919192  |
| BgC       | 0.958414  | 0.868217  |
| DT        | 0.929400  | 0.828283  |

**Best Classifier**

_Random Forest (RF)_
_Achieved the highest combined performance:_

_Accuracy: 0.9758_

_Precision: 0.9829_

_This makes RF the most balanced and reliable classifier in this comparison._

**Notable Mentions**

_Naive Bayes (NB) and KNN scored a perfect Precision of 1.0000._

_NB has strong Accuracy (0.9710) and may be ideal for applications requiring zero false positives._

_KNN, despite perfect precision, has lower accuracy (0.9052) and may struggle with generalization._

_SVC and ETC also deliver excellent results, nearly matching RF._

**Least Performing Classifier**

_Decision Tree (DT)_

_Accuracy: 0.9294_

_Precision: 0.8283 (lowest among all)_

_This model is the least reliable in this analysis due to higher misclassification._

<img width="1178" height="1231" alt="ham_spam_distribution" src="https://github.com/user-attachments/assets/919654ef-460e-494d-b9f8-509f32966967" />

<img width="3017" height="1638" alt="char_distribution" src="https://github.com/user-attachments/assets/e54bead6-15ab-4623-8a96-fb7d9167899b" />

<img width="3017" height="1638" alt="word_distribution" src="https://github.com/user-attachments/assets/57e11df8-e1fb-4373-80fb-c4b78f89e35d" />

<img width="2421" height="2223" alt="pairplot_spam_ham" src="https://github.com/user-attachments/assets/6cb39e3b-a0bc-49a2-874e-bb19e44e49d4" />

<img width="1446" height="1509" alt="wordcloud_spam" src="https://github.com/user-attachments/assets/69dc0d98-78af-4700-9bd0-95047dd21e71" />

<img width="1446" height="1509" alt="wordcloud_ham" src="https://github.com/user-attachments/assets/08bb5250-1aca-4c25-af23-88ec25e73c99" />

<img width="1715" height="1475" alt="top30_spam_words" src="https://github.com/user-attachments/assets/2a620d95-21be-4e2d-8fe7-41628d501590" />

<img width="1715" height="1437" alt="top30_ham_words" src="https://github.com/user-attachments/assets/7f5fad83-e218-4801-b10a-29a4a49b6c74" />

<img width="1256" height="555" alt="Mail_classifier_1" src="https://github.com/user-attachments/assets/ba6cb9ce-774e-4150-880c-f0ed366477f6" />

<img width="916" height="547" alt="Mail_classifier_2" src="https://github.com/user-attachments/assets/14398369-3706-473d-9f8e-e7f6936da505" />

<img width="992" height="523" alt="Mail_classifier_3" src="https://github.com/user-attachments/assets/793f0216-a5e2-4b94-8064-6528edef0db4" />

<img width="1005" height="432" alt="Mail_classifier_4" src="https://github.com/user-attachments/assets/fa742300-393f-473e-afb6-73f11ab45cdb" />

<img width="1250" height="595" alt="Mail_classifier_5" src="https://github.com/user-attachments/assets/9cf3d164-4026-4b10-9d38-cf1dc0e5af22" />


Mail Statistics

<img width="755" height="470" alt="Statictics" src="https://github.com/user-attachments/assets/1cdf53ed-7a85-42a4-aa44-f149856eae7f" />


<img width="1256" height="707" alt="confusionmatrix" src="https://github.com/user-attachments/assets/e93268f5-3dd6-4a80-847e-6da394bb5658" />


Accuracy and Precision of different Algorithms

<img width="1056" height="645" alt="accuracy_precision" src="https://github.com/user-attachments/assets/41766a61-bebb-430d-bde5-a9390a23445b" />

<img width="887" height="532" alt="accuracy_precison_2" src="https://github.com/user-attachments/assets/a1f1ecea-1921-4168-973c-bbecf021c350" />


Overall Performance

<img width="940" height="314" alt="Overall Performance" src="https://github.com/user-attachments/assets/67afcc11-8e05-4246-9c1e-6f5100752d69" />



