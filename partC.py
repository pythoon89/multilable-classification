

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~preprocessing
import re
import nltk
import numpy
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim import *
from gensim import corpora
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest ,chi2



#~~~~~~import data~~~~~~~~~~~
def featuresex(name,title):
    text=[]
    with open(name, 'r') as r:
        for line in r:
            line=line.replace('"','')
#split with , before title column
            line=line[:-1].split(',',139)
            text.append(line)
#~~~~~~~~~~~~~~~~~~~~~
#select useful columns and find their indexes.
    titlename=title
    index=[]
    for item in titlename:
        index.append(text[0].index(item))
        
        trainset=[]
        testset=[]
        index=sorted(index)
        #store all data with objective topics into a newlist "newdata"
        trainset.append([text[0][x] for x in index])
        testset.append([text[0][x] for x in index])
        for i in range(1,len(text)):
            temp=([text[i][x] for x in index])
#if the sum of all value of objecvtive topics is not 0 then this instance is
#appended to my new dataset, the data with 'test' will be stored into 
#testset, and with 'train' will be stored into trainset.
            if sum(list(map(int,temp[3:13])))!=0:
                if temp[2]=='test':
                    testset.append(temp)
                else:
                    trainset.append(temp)
    return trainset,testset





#~~~~~~~~~~~~~~~~~~~~~~~~~~~
def preprocess(dataset):
    newset=[]
    la=[]
    Stopwords = stopwords.words('english')
    # stopwords list
    for j in range(0, len(Stopwords)):
        Stopwords[j] = re.sub('[^0-9a-zA-Z\s]', "", Stopwords[j])
#preprocessing
    for i in range(1,len(dataset)):
        temp=dataset[i]
        temp[-1]=' '.join(temp[-2:])
        temp[-1] = temp[-1].lower()
        temp[-1]=re.sub(r'[a-zA-Z]+://[\S]+|[\S]+[.][\S]+[.][\S]+', '', temp[-1])      
        temp[-1] = re.sub(r'[^a-z 0-9]+', '', temp[-1])
        temp[-1] = re.sub(r'^[0-9]+[\s]|(?<![a-z0-9])[^a-z]+(?![a-z0-9])|[\s][0-9]+$', '', temp[-1])
#        temp[-1] = re.sub(r'^\S{1,2}|[ ][a-z]{1,2}(?=[ ])|(?:[ ])[a-z]{1,2}$', '', temp[-1])
        
        temp[-1] = temp[-1].split()       
        newset.append(temp)
        la.append(temp[3:13])
        
    lemmatizer=WordNetLemmatizer()
    
    final_features = []
    titlefeatures=[]
    
    count = 0
    for item in newset:
        final_features.append([])
        for element in item[-1]:
            element=lemmatizer.lemmatize(element)
            if element not in Stopwords:
                final_features[count].append(element)
        count += 1
    string=[]
    for item in final_features:
        temp=' '.join(item)
        string.append(temp)
    for i in range(1,len(newset)):
        titlefeatures.append(newset[i][-2])
        
    la=numpy.array(la)
    la=la.astype(numpy.float64)
        

        
    return final_features,string,la,titlefeatures

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def naive_bayes_classifier(train_x, train_y):
    from skmultilearn.problem_transform import BinaryRelevance
    from skmultilearn.problem_transform import LabelPowerset
    from skmultilearn.problem_transform import ClassifierChain
    from sklearn.naive_bayes import GaussianNB
    classifier = LabelPowerset(GaussianNB())
#    classifier = ClassifierChain(GaussianNB())
#    classifier = BinaryRelevance(GaussianNB())
    classifier.fit(train_x, train_y)
        
    return classifier 

def logistic_regression_classifier(train_x, train_y):    
    from sklearn.linear_model import LogisticRegression
    from skmultilearn.problem_transform import BinaryRelevance
    from skmultilearn.problem_transform import LabelPowerset
    from skmultilearn.problem_transform import ClassifierChain
    from sklearn.naive_bayes import GaussianNB
    
    model =  LabelPowerset(LogisticRegression(penalty='l1'))
    model.fit(train_x, train_y)    
    return model 

def crossfold(list1,k):    
    length=len(list1)
    k=10
    n=int(length/k)
    traincross=[]
    testcross=[]
    for i in range(k-1):
        if i==0:
            
            testcross.append(numpy.array(list1[i*n:((i+1)*n)]))
            traincross.append(numpy.array(list1[(i+1)*n:]))
        else:
            testcross.append(numpy.array(list1[i*n:((i+1)*n)]))
            temp=numpy.array(list1[0:i*n]+list1[(i+1)*n:])
            traincross.append(temp)

            
    testcross.append(numpy.array(list1[(k-1)*n:]))
    traincross.append(numpy.array(list1[:(k-1)*n]))
    return traincross,testcross

def crossvalid(features,lable,n):

    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import precision_score
    ftrain,ftest=crossfold(features,n)
    ltrain,ltest=crossfold(lable,n)
    result=[[],[]]
    for i in range(len(ftrain)):
        classifier1=naive_bayes_classifier(ftrain[i],ltrain[i])
        classifier2=logistic_regression_classifier(ftrain[i],ltrain[i])
        predictions1 = classifier1.predict(ftest[i])
        predictions2 = classifier2.predict(ftest[i])
        
        result[0].append(accuracy_score(ltest[i],predictions1))
        result[1].append(accuracy_score(ltest[i],predictions2))
    return result


def mlknn(train,label,n):
    from skmultilearn.adapt import MLkNN

    classifier = MLkNN(k=n)
    classifier.fit(train,label)
    return classifier


def confusionmatrix(prediction,la):
    conmatrix=[['TP'],['FN'],['FP'],['TN']]
    for i in range(len(la[0])):
        tp=0;fn=0;fp=0;tn=0
        for n in range(4):
            conmatrix[n].append([])
    
        for j in range(len(prediction)):
            if prediction[j][i]==0 and la[j][i]==0:
                tn=tn+1
            if prediction[j][i]==1 and la[j][i]==1:
                tp=tp+1
            if prediction[j][i]==0 and la[j][i]==1:
                fn=fn+1
            if prediction[j][i]==1 and la[j][i]==0: 
                fp=fp+1
        conmatrix[0][i+1].append(tp);conmatrix[1][i+1].append(fn)
        conmatrix[2][i+1].append(fp);conmatrix[3][i+1].append(tn)
    return conmatrix

    
def evaluations(cmatrix,n):
       
    tp=0
    fp=0
    macroav=0
    for i in range(n):
        tp+=float(cmatrix[0][i+1][0])
        fp+=float(cmatrix[2][i+1][0])
        macroav+=float(nmatrix[0][i+1][0])/(float(cmatrix[0][i+1][0])+float(cmatrix[2][i+1][0]))
    microav=tp/(tp+fp)
    macroav=macroav/n
    return microav,macroav

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
#main function
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#the path of file should be modified before run
title=["pid","fileName","purpose","topic.earn","topic.acq","topic.money.fx","topic.grain","topic.crude","topic.trade","topic.interest","topic.ship","topic.wheat","topic.corn","doc.title","doc.text"]
trainset,testset=featuresex(r'C:\Users\Admin\Desktop\DA\datamining\exercise\excercise2\datamining-exercise2\reutersCSV.csv',title)
set1,strset1,la1,title1=preprocess(trainset)
set2,strset2,la2,title2=preprocess(testset)

from sklearn.feature_extraction.text import TfidfVectorizer
#min_df=4 means the words apear less than 4 times will be ignored
#the vector is a combination form of unigram and bigram
vectorizer = TfidfVectorizer(min_df=4,ngram_range=(1,2))
vectorizer.fit(strset1)
trainvec = vectorizer.transform(strset1)
testvec = vectorizer.transform(strset2)
#features selection kbest~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
kbest=SelectKBest(chi2, k=2000)
kbest.fit(trainvec,la1)
trainvec1=kbest.transform(trainvec)
testvec1=kbest.transform(testvec)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

classifier2=logistic_regression_classifier(trainvec1.toarray(),la1)
classifier1=naive_bayes_classifier(trainvec1.toarray(),la1)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
predictions1 = classifier1.predict(testvec1.toarray())
predictions2 = classifier2.predict(testvec1.toarray())


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

accuracy_score(la2,predictions1)
precision_score(la2,predictions1,average='micro')
precision_score(la2,predictions1,average='macro')
recall_score(la2,predictions1,average='micro')
recall_score(la2,predictions1,average='macro')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

accuracy_score(la2,predictions2)
precision_score(la2,predictions2,average='micro')
precision_score(la2,predictions2,average='macro')
recall_score(la2,predictions2,average='micro')
recall_score(la2,predictions2,average='macro')
classification_report(la2,predictions2)

cv=crossvalid(trainvec1.toarray().tolist(),la1.tolist(),10)
print(cv)   


#
#classifier=mlknn(trainvec1.toarray(),la1,10)
#predictions = classifier.predict(testvec1.toarray())


#
#accuracy=0
#for i in range(10):
#    accuracy+=float((conmatrix[0][i+1][0]+conmatrix[3][i+1][0]))/2548
#    print(float((conmatrix[0][i+1][0]+conmatrix[3][i+1][0]))/2548)
 
        
 
    
    

    


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#vec=CountVectorizer(min_df=3,ngram_range=(1,1),max_features=2000)
#trainvec = vec.fit_transform(strset1)
#testvec=vec.transform(strset2)


#~PCAAAAAAAAAAAAAAAAAAAAAAA
#from sklearn.decomposition import PCA
#pca = PCA(n_components=n2)
#train_f1 = pca.fit_transform(train_f1)
#test_f1 = pca.transform(test_f1)

##~~~~~~~~~~~~~~~~~~~~~~~~~
#
#classifier=naive_bayes_classifier(aa1,bb1)
#predictions = classifier.predict(aa2)
#accuracy_score(bb2,predictions)  



#from sklearn.feature_extraction.text import TfidfVectorizer
#vectorizer = TfidfVectorizer(min_df=3,ngram_range=(1,1),max_features=800)
#vectorizer.fit_transform(corpus)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#dict = corpora.Dictionary(set1)
#corpus = [ dict.doc2bow(text) for text in set1 ]
#texts_tf_idf = models.TfidfModel(corpus)[corpus]
#
#
#lda = LdaModel(corpus=corpus, id2word=dict, num_topics=10) 
#texts_lda = lda[texts_tf_idf]
#
#
#topic_list=lda.print_topics(10)  
#print type(lda.print_topics(10))  
#print len(lda.print_topics(10))  
#print (lda.print_topics(num_topics=3, num_words=4))
#  
#for topic in topic_list:  
#    print topic  
#print "第一主题"  
#print lda.print_topic(1)  
#
#for doc1 in texts_lda:
#    print (doc1)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
#vec=CountVectorizer(min_df=3,ngram_range=(1,1))
#trainvec = vec.fit_transform(strset1)
#testvec=vec.transform(strset2)
#
#from sklearn.feature_selection import VarianceThreshold
#
#sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
#
#sel.fit_transform(trainvec)
#trainvec=sel.fit_transform(trainvec)
#testvec=sel.transform(testvec)
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#from sklearn.decomposition import PCA
#from sklearn.feature_extraction.text import TfidfVectorizer
##vec=CountVectorizer(min_df=3,ngram_range=(1,1),max_features=800)
##vectorizer = TfidfVectorizer(max_df=0.8,max_features=1000)
#vectorizer = TfidfVectorizer(max_df=0.8)
#vectorizer.fit(strset1)
#trainvec = vectorizer.transform(strset1)
#testvec = vectorizer.transform(strset2)
#
#
#
#
#classifier=naive_bayes_classifier(trainvec.toarray(),la1)
#predictions = classifier.predict(testvec.toarray())
#
#from sklearn.metrics import accuracy_score
#accuracy_score(la2,predictions)
#
#
##~~~~~~~~~title tfidi
#from sklearn.decomposition import PCA
#from sklearn.feature_extraction.text import TfidfVectorizer
#vec=CountVectorizer(min_df=3,ngram_range=(1,1),max_features=2500)
##vectorizer = TfidfVectorizer(max_df=0.8)
#vectorizer.fit(title)
#trainvec2 = vectorizer.transform(title1)
#testvec2 = vectorizer.transform(title2)
#
##~~~~~~~~~~~~~~~~mlknn
#from skmultilearn.adapt import MLkNN
#
#classifier = MLkNN(k=10)
#classifier=naive_bayes_classifier(trainvec.toarray(),la1)
#predictions = classifier.predict(testvec.toarray())
#
#from sklearn.metrics import accuracy_score
#accuracy_score(la2,predictions)
# 
# train



