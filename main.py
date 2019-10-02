#import pandas as pd
# import numpy as np
# df = pd.read_csv('avvo_questions_by_practice_area.csv')

#import io
#import codecs
#import csv
#import enchant
#from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize 

#stop_words = set(stopwords.words('english')) 
#english = enchant.Dict("en_US")
#df = pd.read_csv('test.csv', header=None, sep=',')
#lis = [line.split() for line in file]
#line = file1.read() 
#appendFile = open('cleaned_version.csv','a') 
#for r in df: 
#	if not r in stop_words: 
#		if english.check(r):
#			appendFile = open('cleaned_version_1.csv','a') 
#			appendFile.write(" "+r)
#			appendFile.write("\n")
#			appendFile.close()
    
	#appendFile.write("\n")
#appendFile.close()

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from datetime import date
import pandas as pd
import csv
import re
import numpy as np
import random
from nltk.corpus import stopwords

stop = stopwords.words('english')
#df = pd.read_csv("test.csv", header = None, sep=";",encoding='ISO-8859-1')
f=open('avvo.txt',encoding='ISO-8859-1')
reader=csv.reader(f)
df=[]
clear_df=[]
text=[]
label=[]


for i in reader:
	flag=0
	#label.append(i[0])
	t=[]
	t.append(i[0])
	
	t.append(re.sub('[^a-zA-Z\s]','',i[1]))
	
	df.append(t)
	temp=df[-1][1].split()
	clear_df.append(temp)

for i in clear_df:
	count=0
	for j in i:
		if j in stop:
			i.pop(count)
		count+=1


#print(df)
random.shuffle(df)

label=[i[0] for i in df]
text=[i[1] for i in df]

model=TfidfVectorizer(stop_words=stop)
X=model.fit_transform(text)

length=len(df)
num_training=int(length/2)
rs=random.sample(range(length),num_training)
X_train=X[:num_training]
Y_train=label[:num_training]
X_test=X[num_training:]
Y_test=label[num_training:]


print(label)

clf=SVC(kernel='rbf')
clf.fit(X_train,Y_train)

predictions=clf.predict(X_test)

print(predictions)

num = 0
for i, pred in enumerate(predictions):
    if pred == Y_test[i]:
        num += 1
print("precision_score:" + str(float(num)/len(predictions)))




