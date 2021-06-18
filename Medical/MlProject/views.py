from django.shortcuts import render,HttpResponseRedirect,HttpResponse
import json
import pandas as pd
from .jupy import Data
# Create your views here.
def FrontPage(request):
    return render(request,'MlProject/home.html')
def GraphPage(request):
    if request.method == "POST":
        a = request.POST['drop1']
        if a=="1":
            c=Data()
            arr=[]
            feature=['age','bmi','charges']
            for i in feature:
                pp=c.histo(i)
                arr.append(pp)
            h={'p':arr[0],'q':arr[1],'r':arr[2]}
        if a=="2":
            c=Data()
            pp=c.scato()
            h={'q':pp}
        if a=="3":
            c=Data()
            arr=[]
            feature=[['smoker','charges'],['sex','charges'],['region','charges']]
            for i in feature:
                pp=c.baro(i[0],i[1])
                arr.append(pp)
            h={'p':arr[0],'q':arr[1],'r':arr[2]}
        if a=="4":
            c=Data()
            pp=c.corro()
            h={'q':pp}
        return render(request,'MlProject/graphs.html',h)
    else:        
        return render(request,'MlProject/graphs.html')
def PredictPage(request):       
    if request.method == "POST":
        age = request.POST['age']
        sex = request.POST['sex']
        bmi = request.POST['bmi']
        children = request.POST['children']
        smoker = request.POST['smoker']
        region = request.POST['region']
        c=Data()
        pp=c.predict(age,sex,bmi,children,smoker,region)
        lr=pp[0][0]
        lr_score='{0:.3f}'.format(lr[0])
        lr_amount='{0:.2f}'.format(lr[2][0])
        
        rfr=pp[0][1]
        rfr_score='{0:.3f}'.format(rfr[0])
        rfr_amount='{0:.2f}'.format(rfr[2][0])
        
        knn=pp[0][2]
        knn_score='{0:.3f}'.format(knn[0])
        knn_amount='{0:.2f}'.format(knn[2][0])
        
        ridge=pp[0][3]
        ridge_score='{0:.3f}'.format(ridge[0])
        ridge_amount='{0:.2f}'.format(ridge[2][0])
        
        ls=pp[0][4]
        ls_score='{0:.3f}'.format(ls[0])
        ls_amount='{0:.2f}'.format(ls[2][0])
        
        best=pp[1]
        best_score='{0:.3f}'.format(best[0])
        best_amount='{0:.2f}'.format(best[2][0])
        
        pq=c.pie(lr_score,rfr_score,knn_score,ridge_score,ls_score)

        h={"q":pq,"lr_score":lr_score,"lr_amount":lr_amount,"rfr_score":rfr_score,"rfr_amount":rfr_amount,"knn_score":knn_score,"knn_amount":knn_amount,"ridge_score":ridge_score,"ridge_amount":ridge_amount,"ls_score":ls_score,"ls_amount":ls_amount,"best_score":best_score,"best_amount":best_amount,"age":age,"sex":sex,"bmi":bmi,"children":children,"smoker":smoker,"region":region}
        return render(request,'MlProject/amount.html',h)
    else:
        return render(request,'MlProject/predict.html')
def DataPage(request):
    df = pd.read_csv("insurance.csv")
    df1=df.head(50)
    # parsing the DataFrame in json format.
    json_records = df1.reset_index().to_json(orient ='records')
    data = []
    data = json.loads(json_records)
    p = {'d': data}
    return render(request,'MlProject/data.html',p)
def ContactPage(request):
    return render(request,'MlProject/contact.html')
