#! /usr/bin/python
#  Schiff Klassifikation mit Satelliten Daten von Kaggle
#
# Programmstruktur:
# 1. Laden der Daten
#                 
# 2. Organisation der Daten 
#                
# 3. Modellierung
#                 
#                 Preparing of Test and Train Data</li>
#                 Implementation of Artificial Neural Network (ANN)</li>
#                 Implementation of Convolutional Neural Network (CNN)</li>  
#                 < 
# 
#  Daten Informationen
# 
# Planet Daten über der San Francisco Bay von Kaggle: https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery
#    
# Der Datensatz liegt auch als JSON-formatierte Textdatei shipsnet.json vor. Das geladene Objekt enthält Daten, Beschriftungen, Szenen-IDs und Standortlisten.

# Werte sind 1 oder 0, und representieren die "Schiff/Ship" Klassr der "Kein Schiff/no-ship".
# **scene id**: Die eindeutige Kennung der PlanetScope-Szene, aus der der Bildchip extrahiert wurde. Die Szenen-ID kann mit der Planet-API verwendet werden, um die gesamte Szene zu suchen und herunterzuladen.
# **longitude_latitude**: Die Längen- und Breitengradkoordinaten des Bildmittelpunkts, wobei die Werte durch einen einzelnen Unterstrich getrennt sind.

# Die Klasse „Schiff“ umfasst 1000 Bilder. Die Bilder dieser Klasse sind  zentriert auf dem Rumpf eines einzelnen Schiffes. Schiffe unterschiedlicher Größe, Ausrichtung und atmosphärischer Aufnahmebedingungen sind enthalten. Beispielbilder aus dieser Klasse werden unten im Skript angezeigt.

# Die Klasse „kein Schiff“ umfasst 3000 Bilder. Ein Drittel davon ist eine zufällige Auswahl verschiedener Landbedeckungsmerkmale – Wasser, Vegetation, nackte Erde, Gebäude usw. –, die keinen Teil eines Schiffs enthalten. Das nächste Drittel sind „Teilschiffe“, die nur einen Teil eines Schiffs enthalten, aber nicht genug, um die vollständige Definition der Klasse „Schiff“ zu erfüllen. Das letzte Drittel sind Bilder, die zuvor von maschinellen Lernmodellen falsch beschriftet wurden, was normalerweise an hellen Pixeln oder starken linearen Merkmalen lag. Beispielbilder aus dieser Klasse werden unten im Skript angezeigt.

# Zur Installation der Bibliotheken etc siehe auch:
# https://www.activestate.com/resources/quick-reads/how-to-install-keras-and-tensorflow/
#

import numpy as np
from numpy import expand_dims
import pandas as pd
import json
import matplotlib.pyplot as plt
import rasterio
from rasterio import plot

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import tensorflow

from tensorflow.keras.utils import to_categorical

from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

#1. Laden der Daten
# Hier ist ein Besipiel einer Planet Szene..
scene="kaggle/input/ships-in-satellite-imagery/scenes/scenes/lb_2.png"
with rasterio.open(scene) as src:
  data=src.read()
  plot.show(data, title = "Planet Szene 2")

with open('kaggle/input/ships-in-satellite-imagery/shipsnet.json') as data_file:
    dataset = json.load(data_file)
shipsnet= pd.DataFrame(dataset)
shipsnet.head()


#  Wir benötigen nur zwei Spalten: data und labels. 

shipsnet = shipsnet[["data", "labels"]]
shipsnet.head()

len(shipsnet["data"].iloc[0])


# Die Datenwerte von jedem 80x80 RGB Bild sind in einer Liste von 19200 Integer Werten abgelegt. Die ersten 6400 Einträge sind der rote Kanal, die nächten 6400 grün, und 6400 blau.
#

ship_images = shipsnet["labels"].value_counts()[1]
no_ship_images = shipsnet["labels"].value_counts()[0]
print("Anzahl der Schiff-Bilder:{}".format(ship_images),"\n")
print("Anzahl der Kein-Schiff-Bilder:{}".format(no_ship_images))


# 2. Organisation der Daten
# Arrays mit x als y Variablen
x = np.array(dataset['data']).astype('uint8')
y = np.array(dataset['labels']).astype('uint8')

x.shape

# Die aktuellen Daten für jedes Bild sind eine Reihe von abgeflachten 19.200 Datenpunkten, die die RGB-Werte jedes Pixels darstellen. Wir müssen sie also umformen. Nach der Umformung besteht jedes Element in der neuen x-Variable aus 3 Listen. Jede dieser Listen enthält RGB-Werte für jedes Pixel für die Länge und Breite des Bilds.

x_reshaped = x.reshape([-1, 3, 80, 80])
x_reshaped.shape


# Änderung der Dimensionen
x_reshaped = x.reshape([-1, 3, 80, 80]).transpose([0,2,3,1])
x_reshaped.shape
y.shape

#  Die y Variable enthält die Label 1 oder 0. 
y_reshaped = to_categorical(y, num_classes=2)
y_reshaped.shape
y_reshaped

# Wir sehen uns die Bilder an. 
image_no_ship = x_reshaped[y==0]
image_ship = x_reshaped[y==1]

def plot(a,b):
    
    plt.figure(figsize=(15, 15))
    for i, k in enumerate(range(1,9)):
        if i < 4:
            plt.subplot(2,4,k)
            plt.title('Kein Schiff')
            plt.imshow(image_no_ship[i+2])
            plt.axis("off")
        else:
            plt.subplot(2,4,k)
            plt.title('Schiff')
            plt.imshow(image_ship[i+15])
            plt.axis("off")
            
    plt.subplots_adjust(bottom=0.3, top=0.7, hspace=0.25)
    plt.show()

# Anwendung der Funktion 
plot(image_no_ship, image_ship)


# 3. Modelling 
# Normalisierung der X-Daten 

x_reshaped = x_reshaped / 255
#x_reshaped[0][0][0] # Normalisierte RGB Werte des ersten Pixel des ersten Bildes


# Aufteilung der Daten in Training- und Test-Daten.
x_train_1, x_test, y_train_1, y_test = train_test_split(x_reshaped, y_reshaped,
                                                        test_size = 0.20, random_state = 42)

x_train, x_val, y_train, y_val = train_test_split(x_train_1, y_train_1, 
                                                  test_size = 0.25, random_state = 42)

print("x_train shape",x_train.shape)
print("x_test shape",x_test.shape)
print("y_train shape",y_train.shape)
print("y_test shape",y_test.shape)
print("y_train shape",x_val.shape)
print("y_test shape",y_val.shape)

x_train.shape


# Implementation des Convolutional Neural Network (CNN)

from keras import callbacks
model = Sequential()
#
model.add(Conv2D(filters = 64, kernel_size = (4,4),padding = 'Same', 
                 activation ='relu', input_shape = (80,80,3)))
model.add(MaxPool2D(pool_size=(5,5)))
model.add(Dropout(0.25))
#
model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(3,3), strides=(1,1)))
model.add(Dropout(0.25))
#
model.add(Conv2D(filters = 16, kernel_size = (2,2),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(3,3), strides=(1,1)))
model.add(Dropout(0.25))

# Fully connected
model.add(Flatten())
model.add(Dense(200, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(100, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(100, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(50, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(2, activation = "softmax"))

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

earlystopping = callbacks.EarlyStopping(monitor ="val_loss", 
                                        mode ="min", patience = 10, 
                                        restore_best_weights = True)
history = model.fit(x_train, y_train, epochs = 20, validation_data=(x_val, y_val), callbacks = [earlystopping])


model.evaluate(x_test, y_test)


pd.DataFrame(history.history).plot();


#  Daten Augmentation 
# Daten Augmentation ist eine Strategie der Datenerweiterung, die es ermöglicht die Vielfalt der für Trainingsmodelle verfügbaren Daten erheblich zu erhöhen, ohne neue Daten zu sammeln. Datenerweiterungstechniken wie Zuschneiden, Auffüllen und horizontales Spiegeln werden üblicherweise zum Trainieren großer neuronaler Netze verwendet.
datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False, 
        featurewise_std_normalization=False, 
        samplewise_std_normalization=False,  
        zca_whitening=False,
        rotation_range=5,  
        zoom_range = 0.1,
        width_shift_range=0.1,  
        height_shift_range=0.1,  
        horizontal_flip=False, 
        vertical_flip=False)  

datagen.fit(x_train)



# Eine kleine Augmentation auf ein Bild anwenden:

data = x_reshaped[y==1][15]
# Dimension auf eine Datenprobe erweitern
samples = expand_dims(data, 0)
# Bild data augmentation generator
datag = ImageDataGenerator(brightness_range=[0.2,1.0],
                          zoom_range=[0.5,1.0],
                          horizontal_flip=True,
                          rotation_range=90)
# prepare iterator
it = datag.flow(samples, batch_size=1)
# Generiere Daten und Plotte
plt.figure(figsize = (10,10))
for i in range(9):
    plt.subplot(3,3,i+1)
    batch = it.next()
    # Konvertierung zu Integer
    image = batch[0].astype('uint8')
    plt.imshow(image)  
plt.show()

# Daten Augmentation für die training Daten und erneuter fit.

history = model.fit(datagen.flow(x_train, y_train), epochs = 20, 
                    validation_data=(x_val, y_val), callbacks = [earlystopping])


model.evaluate(x_test, y_test)


from sklearn import metrics
import seaborn as sns
Y_pred = model.predict(x_test)

Y_pred_classes = np.argmax(Y_pred,axis = 1) 
Y_true = np.argmax(y_test,axis = 1) 
# Konfusionsmatrix

print("\n""Test Accuracy Score : ",metrics.accuracy_score(Y_true, Y_pred_classes),"\n")

fig, axis = plt.subplots(1, 3, figsize=(20,6))
axis[0].plot(history.history['val_accuracy'], label='val_acc')
axis[0].set_title("Validation Accuracy")
axis[0].set_xlabel("Epochs")
axis[0].set_ylabel("Val. Acc.")
axis[1].plot(history.history['accuracy'], label='acc')
axis[1].set_title("Training Accuracy")
axis[1].set_xlabel("Epochs")
axis[0].set_ylabel("Train. Acc.")
axis[2].plot(history.history['val_loss'], label='val_loss')
axis[2].set_title("Test Loss")
axis[2].set_xlabel("Epochs")
axis[2].set_ylabel("Loss")

plt.show()

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# Plot der Konfusionsmatrix
f,ax = plt.subplots(figsize=(7, 7))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()

plt.show()

pd.DataFrame(history.history).plot();

# Vorhersage der test Data mit dem CNN. Erstes Bild der Test daten

prediction = model.predict(x_test)
pd.Series(prediction[0], index=["Not A Ship", "Ship"])

# Die Geauigkeiten sind gegenüber dem ANN verbesert!
# Schauen wir uns die Bilder einmal an


predicted_data = pd.DataFrame(prediction, columns=["Not a Ship", "Ship"])
predicted_data.head(3)

y_test_data = pd.DataFrame(y_test, columns=["Not a Ship", "Ship"])
y_test_data.head(3)

predicted_data['There is a Ship'] = y_test[:, 1]
predicted_data.head()

predicted_data["Difference"] = predicted_data["Ship"] - predicted_data["There is a Ship"]
predicted_data


# Wenn der Unterschied (Difference) groß ist, bedeutet dies, dass das Bild als Schiff vorhergesagt wurde, obwohl es sich nicht um ein Schiff handelte. Um solche vorhergesagten Bilder zu sehen, müssen wir die Differenzspalte vom größten zum kleinsten sortieren.


predicted_data.sort_values('Difference', ascending=False).head(10)

indexes = predicted_data.sort_values('Difference', ascending = False).head(4).index.to_list()

def plotHistogram(image_index):

    plt.figure(figsize = (10,7))
    plt.subplot(2,2,1)
    plt.imshow(x_test[image_index])
    plt.axis('off')
    plt.title('Kein Schiff, aber vorhergesagt als Schiff.')
    histo = plt.subplot(2,2,2)
    histo.set_ylabel('Count', fontweight = "bold")
    histo.set_xlabel('Pixel Intensity', fontweight = "bold")
    n_bins = 30
    plt.hist(x_test[image_index][:,:,0].flatten(), bins = n_bins, lw = 0, color = 'r', alpha = 0.5);
    plt.hist(x_test[image_index][:,:,1].flatten(), bins = n_bins, lw = 0, color = 'g', alpha = 0.5);
    plt.hist(x_test[image_index][:,:,2].flatten(), bins = n_bins, lw = 0, color = 'b', alpha = 0.5);
    plt.show()


#Implementation der Funktion
for i in indexes:
    plotHistogram(i)

# Die CNN Genauigkeiten sind höher als beim ANN!
#end of CNN code

