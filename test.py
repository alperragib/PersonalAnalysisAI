from keras.models import load_model
import numpy as np
from library import *

model = load_model("C:/Users/alper/PythonProjects/Personal Analysis Project/final_model.h5",custom_objects= {'f1_score': f1_score})
data = np.array([[1, 0.75, 0.5, 0.5, 0.5, 0.5, 0.5]])
predict = model.predict(data)
en_buyuk = predict[0][0]
en_buyuk_indis = 0

for i in range(len(predict[0])):
    if en_buyuk<predict[0][i]:
        en_buyuk=predict[0][i]
        en_buyuk_indis = i

print(en_buyuk),
print(en_buyuk_indis)