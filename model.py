import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import callbacks
from keras import backend as K  
K.set_image_dim_ordering('th')


# load the data
car = np.load('data/car.npy')
circle = np.load('data/circle.npy')
clock = np.load('data/clock.npy')
face = np.load('data/face.npy')
hexagon = np.load('data/hexagon.npy')
octagon = np.load('data/octagon.npy')
panda = np.load('data/panda.npy')
rainbow = np.load('data/rainbow.npy')
smileyface = np.load('data/smileyface.npy')
snowman = np.load('data/snowman.npy')
square = np.load('data/square.npy')
star = np.load('data/star.npy')
triangle = np.load('data/triangle.npy')

'''
print(cat.shape)
print(sheep.shape)
print(giraffe.shape)
print(bat.shape)
print(octopus.shape)
print(camel.shape)
'''

#thêm nhãn
car = np.c_[car, np.zeros(len(car))]
circle = np.c_[circle, np.ones(len(circle))]
clock = np.c_[clock, 2*np.ones(len(clock))]
face = np.c_[face, 3*np.ones(len(face))]
hexagon = np.c_[hexagon, 4*np.ones(len(hexagon))]
octagon = np.c_[octagon, 5*np.ones(len(octagon))]
panda = np.c_[panda, 6*np.ones(len(panda))]
rainbow = np.c_[rainbow, 7*np.ones(len(rainbow))]
smileyface = np.c_[smileyface, 8*np.ones(len(smileyface))]
snowman = np.c_[snowman, 9*np.ones(len(snowman))]
square = np.c_[square, 10*np.ones(len(square))]
star = np.c_[star, 11*np.ones(len(star))]
triangle = np.c_[triangle, 12*np.ones(len(triangle))]





def plot_samples(input_array, rows=4, cols=5, title=''):
    '''
    Hàm để vẽ các bản vẽ 28x28 pixel được lưu trữ trong một mảng numpy.
     Chỉ định số lượng hàng và cột ảnh sẽ hiển thị (4x5 mặc định).
     Nếu mảng chứa ít hình ảnh hơn các ô con đã chọn, các ô con dư thừa vẫn trống.
    '''
    fig, ax = plt.subplots(figsize=(cols,rows))
    ax.axis('off')
    plt.title(title)

    for i in list(range(0, min(len(input_array),(rows*cols)) )):      
        a = fig.add_subplot(rows,cols,i+1)
        imgplot = plt.imshow(input_array[i,:784].reshape((28,28)), cmap='gray_r', interpolation='nearest')
        plt.xticks([])
        plt.yticks([])
        
        
plot_samples(octopus, title='Sample octopus drawings\n')

plot_samples(camel, title='Sample camel drawings\n')


# Hợp nhất các mảng và tách các đối tượng địa lý và nhãn 
X = np.concatenate((car[:10000,:-1], circle[:10000,:-1], clock[:10000,:-1], face[:10000,:-1], hexagon[:10000,:-1], octagon[:10000, :-1], panda[:10000, :-1], rainbow[:10000, :-1], smileyface[:10000, :-1], snowman[:10000, :-1], square[:10000, :-1], star[:10000, :-1], triangle[:10000, :-1]), axis=0).astype('float32') # all columns but the last
y = np.concatenate((car[:10000,-1], circle[:10000,-1], clock[:10000,-1], face[:10000,:-1], hexagon[:10000,-1], octagon[:10000,-1], panda[:10000,-1], rainbow[:10000,-1], smileyface[:10000,-1], snowman[:10000,-1], square[:10000,-1], star[:10000,-1], triangle[:10000,-1]), axis=0).astype('float32') # the last column

#Sau đó, tách dữ liệu giữa tàu và thử nghiệm (tỷ lệ thông thường 80 - 20). Cũng chuẩn hóa giá trị từ 0 đến 1 0 and 1
X_train, X_test, y_train, y_test = train_test_split(X/255.,y,test_size=0.2,random_state=0)


# one hot encode outputs
y_train_cnn = np_utils.to_categorical(y_train)
y_test_cnn = np_utils.to_categorical(y_test)
num_classes = y_test_cnn.shape[1]
# reshape to be [samples][pixels][width][height]
X_train_cnn = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test_cnn = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')



def cnn_model():
    # create model
    model = Sequential()
    model.add(Conv2D(30, (3, 3), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2)),
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

np.random.seed(0)
# build the model
model_cnn = cnn_model()
# Fit the model
model_cnn.fit(X_train_cnn, y_train_cnn, validation_data=(X_test_cnn, y_test_cnn), epochs=1, batch_size=200)
# Final evaluation of the model
scores = model_cnn.evaluate(X_test_cnn, y_test_cnn, verbose=0)
print('Final CNN accuracy: ', scores[1])




y_pred_cnn = model_cnn.predict_classes(X_test_cnn, verbose=0)
from sklearn import metrics
c_matrix = metrics.confusion_matrix(y_test, y_pred_cnn)
import seaborn as sns
def confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
class_names = ['car', 'circle', 'clock', 'face', 'hexagon', 'octagon', 'panda', 'rainbow', 'smileyface', 'snowman', 'square', 'star', 'triangle']
confusion_matrix(c_matrix, class_names, figsize = (10,7), fontsize=14)

#Misclassification when y_pred and y_test are different.
misclassified = X_test[y_pred_cnn != y_test]
plot_samples(misclassified, rows=10, cols=5, title='')


target_dir = './models/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
model_cnn.save('./models/model.h5')
model_cnn.save_weights('./models/weights.h5')
