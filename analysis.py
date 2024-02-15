#data available here: https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

data = pd.read_csv("apple_quality.csv")
print(data)
dataUse = data.drop(['A_id'], axis=1)
dataUse['Quality'] = dataUse['Quality'].map({'good': 1, 'bad': 0})
print(dataUse)
print(dataUse.isnull().sum())
dataUse = dataUse.dropna()
print(dataUse.isnull().sum())
from sklearn.model_selection import train_test_split
X = dataUse.drop(['Quality'], axis=1).values
y = dataUse[['Quality']].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)

from sklearn.preprocessing import RobustScaler,StandardScaler

scaler = StandardScaler()
robust = RobustScaler()

X_trainScaled = robust.fit_transform(X_train)
X_testScaled = robust.fit_transform(X_test)

X_trainScaled = scaler.fit_transform(X_trainScaled)
X_testScaled = scaler.fit_transform(X_testScaled)

import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get('accuracy') > 0.95 and logs.get('val_accuracy') > 0.95:
                print("\nReached 95% accuracy so cancelling training!")

                self.model.stop_training = True
myCallback = myCallback()

model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=128, input_shape=[7],activation='relu'),
            tf.keras.layers.Dense(units=128,activation='relu'),
            tf.keras.layers.Dense(units=1, activation='sigmoid')
        ])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_trainScaled, y_train,validation_data=(X_testScaled, y_test), epochs=50,verbose=1, callbacks=[myCallback])

print(history)

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left') 
plt.show()

