import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras import backend as K
from matplotlib import pyplot as plt

data = pd.read_csv("C:/Users/alper/PythonProjects/Personal Analysis Project/psyc.csv")

gender = {'Male': 1,'Female': 0}
data.gender = [gender[item] for item in data.gender]

# dependable: güvenilir, serious: ciddi, responsible: sorumluluk sahibi, extraverted: dışa dönük, lively: enerjik
personality = {'dependable': 0,'serious': 1,'responsible': 2,'extraverted': 3,'lively': 4}
data.Personality = [personality[item] for item in data.Personality]

# openness: açık sözlülük, neuroticism: kararsızlık, conscientiousness: dürüstlük, agreeableness: uyumluluk, extraversion: dışa dönüklük 
data.openness = [item/10 for item in data.openness]
data.neuroticism = [item/10 for item in data.neuroticism]
data.conscientiousness = [item/10 for item in data.conscientiousness]
data.agreeableness = [item/10 for item in data.agreeableness]
data.extraversion = [item/10 for item in data.extraversion]

maks = max(data.age)
mini = min(data.age)

max_min_norm = lambda v: (v-mini)/(maks-mini)

data.age = [max_min_norm(item) for item in data.age]

y = data.Personality
x = data.drop(['Personality'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1,random_state=24)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

model = Sequential()
model.add(Dense(7,input_shape=(x_train.shape[1],), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(16, activation='relu'))
model.add(Dense(5,activation="softmax"))
model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

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
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


model.compile(loss="categorical_crossentropy",optimizer=optimizer,metrics=["acc",f1_score])
history = model.fit(x_train,y_train,validation_data=(x_test,y_test),validation_split=0.2,epochs=500)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Başarımı')
plt.ylabel('Başarım')
plt.xlabel('Epoch')
plt.legend(['Eğitim', 'Doğrulama'], loc='upper left')
plt.show()

#model.save("final_model.h5")

loss, accuracy, f1_score = model.evaluate(x_test, y_test)
print(accuracy,f1_score)

# accuracy=0.7  loss=0.3  f1_score=0.68