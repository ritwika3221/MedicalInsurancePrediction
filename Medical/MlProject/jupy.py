import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from io import StringIO
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV

class Data:
    df=pd.read_csv(r'insurance.csv')
    def histo(self,i):
        df1=self.df
        fig=plt.figure(figsize=(4,4))
        df2=self.df[i]
        sns.distplot(df2,kde=False,hist_kws={'color':'r','edgecolor':'b','linewidth':3,'alpha':0.7})
        plt.title(str(i.upper())+' Distribution')
        plt.ylabel('Count')
        plt.xticks(rotation=90)
        plt.tight_layout()
        imgdata=StringIO()
        fig.savefig(imgdata,format='svg')
        imgdata.seek(0)
        data=imgdata.getvalue()
        print(df2)
        return data
    
    def scato(self):
        df1=self.df
        fig=plt.figure(figsize=(4,4))
        sns.scatterplot(x=df1['bmi'],y=df1['charges'],hue=df1['sex'],palette='Reds')
        plt.title('BMI VS Charge')
        imgdata=StringIO()
        fig.savefig(imgdata,format='svg')
        imgdata.seek(0)
        data=imgdata.getvalue()
        return data
    def baro(self,x,y):
        df1=self.df
        fig=plt.figure(figsize=(3,5))
        df2=self.df[x]
        df3=self.df[y]
        plt.title(str(x) +" vs Charge")
        sns.barplot(x=df2,y=df3,palette='Set3')
        plt.xticks(rotation=90)
        plt.tight_layout()
        imgdata=StringIO()
        fig.savefig(imgdata,format='svg')
        imgdata.seek(0)
        data=imgdata.getvalue()
        return data
    def corro(self):
        # clean_data = {'sex': {'male' : 0 , 'female' : 1} ,
        #          'smoker': {'no': 0 , 'yes' : 1},
        #            'region' : {'northwest':0, 'northeast':1,'southeast':2,'southwest':3}
        #        }
        df1=self.df
        # df1.replace(clean_data,inplace=True)
        fig,ax=plt.subplots(figsize=(5,5))
        sns.heatmap(df1.corr(),cmap='BuPu',annot=True,fmt=".2f",ax=ax)
        plt.title("Dependencies of Medical Charges")
        imgdata=StringIO()
        fig.savefig(imgdata,format='svg')
        imgdata.seek(0)
        data=imgdata.getvalue()
        return data

    def algo(self,x,y,p):
        a=[]
        xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=40)
        m1=LinearRegression()
        m1.fit(xtrain,ytrain)
        pre1=m1.predict(p)
        a.append([m1.score(xtrain,ytrain),"LR",pre1])
        

        m2=RandomForestRegressor()
        # parameters = { 'n_estimators':[600,1000,1200],
        #      'max_features': ["auto"],
        #      'max_depth':[40,50,60],
        #      'min_samples_split': [5,7,9],
        #      'min_samples_leaf': [7,10,12],
        #      'criterion': ['mse']}
        # reg_rf_gscv = GridSearchCV(estimator=m2, param_grid=parameters, cv=10, n_jobs=-1)
        m2.fit(xtrain,ytrain)
        pre2=m2.predict(p)
        a.append([m2.score(xtrain,ytrain),"RFR",pre2])

        param={'n_neighbors':[2,3,4,5,6,7,8,9]}
        mm3=neighbors.KNeighborsRegressor()
        m3=GridSearchCV(mm3,param,cv=5)
        m3.fit(xtrain,ytrain)
        pre3=m3.predict(p)
        a.append([m3.score(xtrain,ytrain),"Kneighbours",pre3])

        m4=Ridge(alpha=50,max_iter=100,tol=0.1)
        # parameters = { 'model__alpha': [1e-15, 1e-10, 1e-8, 1e-3, 1e-2,1,2,5,10,20,25,35, 43,55,100], 'model__random_state' : [42]}
        # reg_ridge = GridSearchCV(estimator=m4, param_grid=parameters, cv=10)
        m4.fit(xtrain,ytrain)
        pre4=m4.predict(p)
        a.append([m4.score(xtrain,ytrain),"Ridge",pre4])
        
        m5=Lasso(alpha=50,max_iter=100,tol=0.1)
        m5.fit(xtrain,ytrain)
        pre5=m5.predict(p)
        a.append([m5.score(xtrain,ytrain),"Ridge",pre5])
        return a

    def predict(self,age,sex,bmi,children,smoker,region):
        df1=self.df
        df2=pd.DataFrame({"age":[age],
        "sex":[sex],
        "bmi":[bmi],
        "children":[children],
        "smoker":[smoker],
        "region":[region],

        })

        # print(df2)
        df10=df1.copy()
        df10.replace([np.inf,-np.inf],np.nan,inplace=True)
        df4=df10.dropna()        
        df22=df4[["age","sex","bmi","children","smoker","region"]]
        df7=df4.charges
        df11=df22.append(df2,ignore_index=True)
        df13=df11.copy()
        
        a=LabelEncoder()
        df13['sex_n']=a.fit_transform(df13['sex'])
        df13['smoker_n']=a.fit_transform(df13['smoker'])
        df13['region_n']=a.fit_transform(df13['region'])
        df5=df13.drop(['sex','smoker','region'],axis="columns")

        df99=df5.iloc[[-1]]
        print(df99)
        df5.drop(df5.tail(1).index,inplace=True)
        pp=self.algo(df5,df7,df99)
        maxi=0.0
        ind=0
        for i in range(0,len(pp)):
            z=pp[i][0]
            if z>maxi:
                maxi=z
                ind=i
        dq=[pp,pp[ind]]
        return dq
        
    def pie(self,lr_score,rfr_score,knn_score,ridge_score,ls_score):
        score=[lr_score,rfr_score,knn_score,ridge_score,ls_score]
        score_name=["Linear","Random_Forest","K-Neighbors","Ridge","Lasso"]
        fig=plt.figure(figsize=(5,5))
        plt.pie(score,autopct='%1.1f%%',labels=score_name,shadow=True)
        imgdata=StringIO()
        fig.savefig(imgdata,format='svg')
        imgdata.seek(0)
        data=imgdata.getvalue()
        return data        