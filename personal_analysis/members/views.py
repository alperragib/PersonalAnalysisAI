from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from django.urls import reverse
from keras.models import load_model
import numpy as np
from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_score(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)

def index(request):

  sorular = [
    {"indis":1,"soru":"İnsanlar sık sık söylediğim şeylerden gerçekten alınabiliyor."},
    {"indis":2,"soru":"Bana verilen görevlere öfkelenirim. Verilen işleri yaparken ağırdan alırım."},
    {"indis":3,"soru":"Öz eleştiri yaparım ve kusurlu yanlarını kabul edip düzeltmeye çalışırım."},
    {"indis":4,"soru":"Her zaman karşımdaki kişi ile konuşurken empati kurarım."},
    {"indis":5,"soru":"Sık sık seyahat ederim. Farklı şehirleri ve ülkeleri keşfetmekten hoşlanırım."}, 
    {"indis":6,"soru":"Ne düşündüğümü söylemekten hiçbir zaman korkmam."},
    {"indis":7,"soru":"Israrcı biçimde dediğimin kabul edilmesini ve yapılmasını isterim."},
    {"indis":8,"soru":"Zor durumda kalsam bile asla yalan söylemem."},
    {"indis":9,"soru":"Birisi konuşuyorsa sözünü asla kesmem konuşmasının bitmesini beklerim."},
    {"indis":10,"soru":"Genelde evde oturmak yerine dışarıda olmayı tercih ederim."}, 
  ]

  template_home = loader.get_template('home_page.html')
  context_home = {
    'sorular': sorular,
  }

  template_sonuc = loader.get_template('sonuc_page.html')
  
  try:
    age = int(request.POST['age'])
    gender = request.POST['gender']
    bir = int(request.POST['1'])
    iki = int(request.POST['2'])
    uc = int(request.POST['3'])
    dort = int(request.POST['4'])
    bes = int(request.POST['5'])
    alti = int(request.POST['6'])
    yedi = int(request.POST['7'])
    sekiz = int(request.POST['8'])
    dokuz = int(request.POST['9'])
    on = int(request.POST['10'])

    maks = 28
    mini = 5

    if age>maks:
        maks = age
    if age<mini:
        mini = age
    
    age = (age-mini)/(maks-mini)

    if gender=='male':
        gender = 1
    else:
        gender = 0


    openness = (bir+alti)/10
    neuroticism = (iki+yedi)/10
    conscientiousness = (uc+sekiz)/10
    agreeableness = (dort+dokuz)/10
    extraversion = (bes+on)/10

    model = load_model("C:/Users/alper/PythonProjects/Personal Analysis Project/final_model.h5",custom_objects= {'f1_score': f1_score})
    data = np.array([[gender, age, openness, neuroticism, conscientiousness, agreeableness, extraversion]])
    predict = model.predict(data)
    en_buyuk = predict[0][0]
    en_buyuk_indis = 0
    for i in range(len(predict[0])):
        if en_buyuk<predict[0][i]:
            en_buyuk=predict[0][i]
            en_buyuk_indis = i

    sonuc = ["Güvenilir (Dependable)", "Ciddi (Serious)", "Sorumluluk sahibi (Responsible)", "Dışa dönük (Extraverted)", "Enerjik (Lively)"]

    context_sonuc = {
     'sonuc': sonuc[en_buyuk_indis],
    }


    return HttpResponse(template_sonuc.render(context_sonuc, request))
  except:
    return HttpResponse(template_home.render(context_home, request))
    
