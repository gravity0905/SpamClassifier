# SpamClassifier
The objective of this project is to compare the performance of two popular Naive Bayes Spam Classifiers
+ Multi-variate Bernoulli event model
+ Multinomial event model

### Spam Dataset
The [Ling-Spam](http://csmining.org/index.php/ling-spam-datasets.html) corpus is used for training the models.

### Preprocessing
All the mails in the `bare` subdirectory were preprocessed using the `process.py` script and stored in another directory. 
The following email preprocessing and normalization steps were carried out in the given order:
+ Lower casing
+ Stripping HTML tags
+ Normalizing URLs
+ Normalizing email addresses
+ Normalizing numbers
+ Normalizing currency symbols
+ Removal of non-word characters
+ Stop Word removal
+ Word Stemming

### Word Stemming
The [Porter Stemming algorithm](https://tartarus.org/martin/PorterStemmer/) which was ported to Python from the
version coded up in ANSI C by the author was used for word stemming.

### License
Copyright (c) 2017 Garvit Aggarwal
