import numpy as np
import pandas as pd
import math
from scipy.stats import multivariate_normal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, precision_score,balanced_accuracy_score, jaccard_score
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec




model=RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
"""
#Multivariate Gaussian Distribution Manually
def pr(x,mu,std):
    s=(1/(math.sqrt(2*math.pi)*std))*math.exp((-1)*(1/(2*std*std))*((x-mu)**2))
    return s
def est(d,mu,sigma):
    p=multivariate_normal(mean=mu,cov=sigma)

    return p.pdf(d)
def norm(d):
    mu=np.mean(d,axis=0)
    std=np.std(d,axis=0)
    return (d-mu)/std
"""
A=pd.read_csv('Training data.csv')


C=A['risk_flag']

B=A.drop(columns=['Id','city','profession','state'],axis=1)
B=pd.get_dummies(B,drop_first=True)
B['house']=B['house_ownership_rented'].values+B['house_ownership_owned'].values
B=B.drop(columns=['house_ownership_rented','house_ownership_owned'],axis=1)

B1=B.drop(columns=['house','married_single','risk_flag'],axis=1)
#model.fit(B1,C)
#f = model.feature_importances_

#Parameters- 'income', 'age', 'experience', 'current_job_years',
#       'current_house_years', 'car_ownership_yes'
gen=B[B['risk_flag']==0]

ana=B[B['risk_flag']==1]

gen=gen.drop(columns=['house','married_single'])
ana=ana.drop(columns=['house','married_single'])
g=len(gen)
a=len(ana)
gen1=gen.copy()
ana1=ana.copy()

#Relation between income,age and risk_flag
plt.scatter(gen1[:200]['income'],gen1[:200]['age'],color='b',s=65)
plt.scatter(ana1[:200]['income'],ana1[:200]['age'],color='r',s=65)
plt.show()
    
X_cv=pd.concat([gen[g//2:int(g//1.33)],ana[:int(a//2)]],axis=0)
X_test=pd.concat([gen[int(g//3):],ana[int(a//2):]],axis=0)
y_cv1=X_cv['risk_flag']
y_test=X_test['risk_flag']
X_cv=X_cv.drop(columns=['risk_flag'])
X_test=X_test.drop(columns=['risk_flag'])
y_cv=y_cv1.map({1:True,0:False}).values
y_test=y_test.map({1:True,0:False}).values


# Fitting data onto a gaussian distribution using flag=0 data.
# Training set= Genuine Set
# Crossvalidation Set= 1/4th genuine + 1/2 anamoly class
# Test Set= 1/4th genuine + 1/2 anamoly class

gen=gen.drop(columns=['risk_flag'])
ana=ana.drop(columns=['risk_flag'])
gen=(gen-np.mean(gen))/np.std(gen)
X_cv=(X_cv-np.mean(X_cv))/np.std(X_cv)
X_test=(X_test-np.mean(X_test))/np.std(X_test)

# Using mulivariate_normal to find Probability Density Function

mu=np.mean(gen)
sigma=np.cov(gen.T)
p1=multivariate_normal(mu,sigma)
p=p1.pdf(X_cv)

a=float(max(p))
b=float(min(p))
i=a
i1=0
m=0

ctr=0
# Running loop to find appropriate value of cutoff probability by maximising f1
# score
while(True):
    i=i-(a-b)/1000
    
    preds=(p<i)
    y1=f1_score(y_cv,preds)
    
    x=y1
    if(x>m):
        m=x
        i1=i
        pr=preds
    
    if(i<b):
        break

print('Precision Score for CV:',precision_score(y_cv,pr))
print('Recall Score for CV:',recall_score(y_cv,pr))
print('F1 Score for CV',f1_score(y_cv,pr))
print()

p2=p1.pdf(X_test)
preds1=p2<i1

print('Precision Score for Test Set:',precision_score(y_test,preds1))
print('Recall Score for Test Set:',recall_score(y_test,preds1))
print('F1 Score for Test Set:',f1_score(y_test,preds1))
#Risk_Flag seems arbitrary and it is difficult to derive any inference from
# the main factors (income and age)
