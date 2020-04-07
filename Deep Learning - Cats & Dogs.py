import os
import shutil
from os import path
import keras

from keras import models
from keras import layers

from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

origin_folder = "C:\\Users\\b10077\\Desktop\\Python\\Cats & Dogs\\Original"

base_folder = "C:\\Users\\b10077\\Desktop\\Python\\Cats & Dogs\\Small"

if (path.isdir(base_folder) == False):
    
    os.mkdir(base_folder)
    
    train_folder = os.path.join(base_folder,'train')
    os.mkdir(train_folder)
    
    validation_folder = os.path.join(base_folder,'validation')
    os.mkdir(validation_folder)
    
    test_folder = os.path.join(base_folder,'test')
    os.mkdir(test_folder)
    
    
    train_cats_folder = os.path.join(train_folder,'cats')
    os.mkdir(train_cats_folder)
    
    train_dogs_folder = os.path.join(train_folder,'dogs')
    os.mkdir(train_dogs_folder)
    
    validation_cats_folder = os.path.join(validation_folder,'cats')
    os.mkdir(validation_cats_folder)
    
    validation_dogs_folder = os.path.join(validation_folder,'dogs')
    os.mkdir(validation_dogs_folder)
    
    test_cats_folder = os.path.join(test_folder,'cats')
    os.mkdir(test_cats_folder)
    
    test_dogs_folder = os.path.join(test_folder,'dogs')
    os.mkdir(test_dogs_folder)
    
    fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src=os.path.join(origin_folder, fname)
        dst = os.path.join(train_cats_folder, fname)
        shutil.copyfile(src,dst)
        
    fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src=os.path.join(origin_folder, fname)
        dst = os.path.join(validation_cats_folder, fname)
        shutil.copyfile(src,dst)
        
    fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src=os.path.join(origin_folder, fname)
        dst = os.path.join(test_cats_folder, fname)
        shutil.copyfile(src,dst)
        
        
    fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src=os.path.join(origin_folder, fname)
        dst = os.path.join(train_dogs_folder, fname)
        shutil.copyfile(src,dst)
        
    fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src=os.path.join(origin_folder, fname)
        dst = os.path.join(validation_dogs_folder, fname)
        shutil.copyfile(src,dst)
        
    fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src=os.path.join(origin_folder, fname)
        dst = os.path.join(test_dogs_folder, fname)
        shutil.copyfile(src,dst)
        
#### Model creation without augmentation
        
        

#model = models.Sequential()
#model.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape=(150,150,3)))
#model.add(layers.MaxPooling2D(2,2))
#model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
#model.add(layers.MaxPooling2D(2,2))
#model.add(layers.Conv2D(128, (3,3), activation = 'relu'))
#model.add(layers.MaxPooling2D(2,2))
#model.add(layers.Conv2D(128, (3,3), activation = 'relu'))
#model.add(layers.MaxPooling2D(2,2))
#model.add(layers.Flatten())
#model.add(layers.Dense(512, activation = 'relu'))
#model.add(layers.Dense(1,activation = 'sigmoid'))
#print(model.summary())
#
#model.compile(optimizer = 'rmsprop',
#               loss = 'binary_crossentropy',
#               metrics = ['accuracy'])
# 
# 
# 
#train_datagen = ImageDataGenerator(rescale = 1./255)
#test_datagen = ImageDataGenerator(rescale = 1./255)
#
#train_generator = train_datagen.flow_from_directory(train_folder,target_size=(150,150),
#                                                    batch_size = 20,class_mode = 'binary')
#validation_generator = test_datagen.flow_from_directory(validation_folder,target_size=(150,150),
#                                                    batch_size = 20,class_mode = 'binary')
#for data_batch, labels_batch in train_generator:
#     print('data batch shape:', data_batch.shape)
#     print('labels batch shape:', labels_batch.shape)
#     break
# 
#history = model.fit_generator(train_generator,steps_per_epoch = 100, epochs = 30
#                               , validation_data = validation_generator, validation_steps = 50)
#model.save('cats_and_dogs_small_2.h5')

## Model with Dataaugmentation
        
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=40,width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_folder,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_folder,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')


history = model.fit_generator(
train_generator,
steps_per_epoch=100,
epochs=100,
validation_data=validation_generator,
validation_steps=50)
model.save('cats_and_dogs_small_2.h5')


model1 = load_model('cats_and_dogs_small_2.h5')
predict_datagen = ImageDataGenerator(rescale=1./255,rotation_range=40,width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,)
#model.summary()
predict_image = image.load_img("C:\\Users\\b10077\\Desktop\\Python\\Cats & Dogs\\Predict\\cat.jpg", target_size=(150, 150))
predict_image = image.img_to_array(predict_image)
predict_image = np.expand_dims(predict_image, axis=0)
result = model1.predict_classes(predict_image, batch_size=1)
print(result)


model1 = load_model('cats_and_dogs_small_2.h5')
batch_holder = np.zeros((17, 150, 150,3))
img_dir='C:\\Users\\b10077\\Desktop\\Python\\Cats & Dogs\\Predict'
for i,img in enumerate(os.listdir(img_dir)):
  img = image.load_img(os.path.join(img_dir,img), target_size=(150,150))
  batch_holder[i, :] = img

result=model1.predict_classes(batch_holder)
 
fig = plt.figure(figsize=(20, 20))
 
for i,img in enumerate(batch_holder):
  fig.add_subplot(4,5, i+1)
  if (result[i][0] == 1):
      plt.title('Dog')
  else:
      plt.title('Cat')
  plt.imshow(img/256.)
  
plt.show()

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()