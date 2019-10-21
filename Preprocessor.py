import numpy as np
import copy
import csv
from tqdm import tqdm
import imp
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import gensim.downloader as api
from os import listdir

class DataReader:

    def __init__(self,file_path,sub_task=None):
        self.file_path = file_path
        self.sub_task = sub_task

    def get_labelled_data(self):
        data = []
        labels = []
        with open(self.file_path,encoding='utf8') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for i,line in enumerate(tqdm(reader,'Reading Data')):
                if i is 0:
                    continue
                label = self.str_to_label(line[-3:])
                if  self.sub_task:
                    self.filter_subtask(data,labels,line[1],label)
                else:
                    labels.append(label)
                    data.append(line[1])
        return data,labels
    
    def get_test_data(self):
        data = []
        ids = []
        with open(self.file_path,encoding='utf8') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for i,line in enumerate(tqdm(reader,'Reading Test Data')):
                if i is 0:
                    continue
                ids.append(line[0])
                data.append(line[1])
        return data,ids
    
    def str_to_label(self,all_labels):
        label = 0
        if all_labels[0] == 'OFF':
            if all_labels[1] == 'UNT':
                label = 1
            elif all_labels[1] == 'TIN':
                if all_labels[2] == 'IND':
                    label = 2
                elif all_labels[2] == 'GRP':
                    label = 3
                elif all_labels[2] =='OTH':
                    label = 4
        return label
    
    def filter_subtask(self,data,labels,sample,label):
        if self.sub_task == 'A':
            data.append(sample)
            labels.append(int(label>0))
        elif self.sub_task =='B':
            if label > 0:
                data.append(sample)
                labels.append(int(label>1))
        elif self.sub_task == 'C':
            if label > 1:
                data.append(sample)
                labels.append(label-2)

class Preprocessor:
    def __init__(self,*args):
        self.params =[]
        if args:
            if isinstance(args[0],tuple):
                self.params = list(*args)
            else:
                self.params = list(args)
        self.params = ['tokenize']+self.params

    def tokenize(self):
        from nltk import word_tokenize
        for i,tweet in tqdm(enumerate(self.data),'Tokenization'):
            self.data[i] = word_tokenize(tweet.lower())
        return self.data

    def remove_stopwords(self):
        from nltk.corpus import stopwords
        import re
        stop = set(stopwords.words("english"))
        noise = ['user']
        for i,tweet in tqdm(enumerate(self.data),'Stopwords Removal'):
            self.data[i] = [w for w in tweet if w not in stop and not re.match(r"[^a-zA-Z\d\s]+", w) and w not in noise]
        return self.data
    
    def get_pos(self, word):
        from nltk import pos_tag
        from nltk.corpus import wordnet
        tag = pos_tag([word])[0][1]
        if tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def lemmatize(self):
        from nltk.stem import WordNetLemmatizer
        wnl = WordNetLemmatizer()
        for i, tweet in tqdm(enumerate(self.data),'Lemmatization'):
            for j, word in enumerate(tweet):
                self.data[i][j] = wnl.lemmatize(word, pos=self.get_pos(word))
        return self.data
    
    def stem(self):
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
        for i,tweet in tqdm(enumerate(self.data),'Stemming'):
            for j,word in enumerate(tweet):
                self.data[i][j] = stemmer.stem(word)
        return self.data

    def clean(self, data):
        self.data = copy.deepcopy(data)
        for param in tqdm(self.params,'Preprocessing'):
            clean_call = getattr(self, param,None)
            if clean_call:
                clean_call()
            else:
                raise Exception(str(param)+' is not an available function')
        return self.data
    
class Vectorizer:
    def __init__(self,type,pre_trained=False,retrain=False,extend_training=False,params={}):
        self.type = type
        self.pre_trained = pre_trained
        self.params = params
        self.retrain = retrain
        self.extend_training = extend_training
        self.vectorizer = None
        self.max_len = None

    def glove(self):
        print('\nLoading Glove Embeddings from api...')
        model = api.load('glove-twitter-100')
        vectorizer = model.wv
        vectors = [np.array([vectorizer[word] for word in tweet if word in model]).flatten() for tweet in tqdm(self.data,'Vectorizing')]
        self.vocab_length = len(model.wv.vocab)
        if not self.max_len:
            self.max_len = np.max([len(vector) for vector in vectors])
        self.vectors = [
            np.array(vector.tolist()+[0 for _ in range(self.max_len-len(vector))]) for vector in tqdm(vectors,'Finalizing')
            ]
        for i,vec in enumerate(self.vectors):
            self.vectors[i] = vec[:self.max_len]
        return self.vectors

    def vectorize(self,data):
        self.data = data
        vectorize_call = getattr(self, self.type, None)
        if vectorize_call:
            vectorize_call()
        else:
            raise Exception(str(self.type),'is not an available function')
        return self.vectors
    
    def fit(self,data):
        self.data = data
        
class Classifier:
    def __init__(self,type,params={}):
        __classifers__ = {
        'SVC': SVC
        }
        self.classifier = __classifers__[type]
        self.params = params
        self.model = self.classifier(**self.params)   

    def fit(self,X_train,Y_train):
        return self.model.fit(X_train,Y_train)

    def predict(self,X_test):
        return self.model.predict(X_test)

    def score(self, X_train, Y_train, X_test, Y_test):
        train_pred = self.model.predict(X_train)
        print("Training Data Score:", accuracy_score(Y_train, train_pred))
        predictions = self.model.predict(X_test)
        print("Test Data Score:", accuracy_score(Y_test, predictions))
        print("Precision: ", precision_score(Y_test, predictions, average=micro))
        print("Recall: ", recall_score(Y_test, predictions, average=micro))
        print("F1 Score", f1_score(Y_test, predictions, average=micro))
        print("Confusion matrix", confusion_matrix(Y_test, predictions))

    def tune(self,X_train,Y_train,tune_params=None,best_only=False,scoring='f1'):
        if not tune_params:
            tune_params = self.params
        tuner = GridSearchCV(self.model,tune_params,n_jobs=4,verbose=1,scoring=scoring)
        tuner.fit(X_train,Y_train)
        self.model = tuner.best_estimator_
        if best_only:
            return {'score':tuner.best_score_,'params':tuner.best_params_}
        else:
            param_scores = {}
            results = tuner.cv_results_
            for i,param in enumerate(tuner.cv_results_['params']):
                param_str  = ', '.join("{!s}={!r}".format(key,val) for (key,val) in param.items())
                param_scores[param_str]={'test_score':results['mean_test_score'][i],'train_score':results['mean_train_score'][i]}
            return param_scores
    
    def get_model(self):
        if getattr(self,'model',None):
            return self.model
        else:
            raise Exception('Model has not been created yet.')