from keras.models import Sequential
from keras.layers.core import Dense,Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layer.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2, activity_l2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import math
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os
import csv
import tesnsorflow as tf
tf.python.control_flow_ops = tf



def process_dispimage(image, angle, pred_angle, frame):
    """
    Data visualization for image data
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
    img = cv2.resize(img,None,fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    h,w = img.shape[0:2]

    cv2.putText(img, 'frame:' + str(frame), org=(2,18), fontFace=font, fontScale=.5, color=(200,100,100), thickness=1)
    cv2.putText(img, 'angle:' + str(angle), org=(2,33), fontFace=font, fontScale=.5, color=(200,100,100), thickness=1)

    cv2.line(img,(int(w/2),int(h)),(int(w/2+angle*w/4),int(h/2)),(0,255,0), thickness=4)
    if pred_angle is not None:
        cv2.line(img,(int(w/2),int(h)),(int(w/2+angle*w/4),int(h/2)),(0,0,255,thickness=4))
    return img

def visualize_dataset(X,y,y_pred=None):
    """
    Format the data from the dataset and display
    """
    for i in range(len(X)):
        if y_pred is not None:
            img = process_dispimage(X[i],y[i],y_pred[i],i)
        else:
            img = process_dispimage(X[i],y[i],y_pred[i],i)
        cv2.imshow("image",img)
        cv2.waitkey(0)
        cv2.destryoAllWindows()

def preprocess_image(img):
    """
    Function to preprocess images before applying it to the NN
    """
    crop_img = img[50:150,:,:]
    blur_img = cv2.GaussianBlur(crop_img, (3,3),0)
    resize_img = cv2.resize(blur_img,(200,66), interpolation=cv2.INTER_AREA)
    yuv_img = cv2.cvtColor(resize_img, cv2.COLOR_BGRYUV)
    return yuv_img


def distort_img(img,angle):
    """
    Adding random distortion to the images like brightness adjust,
    random vertical shifts
    """
    dis_img = img.astype(float)
    val = np.random.randint(-28,28)
    if val > 0:
        mask = (dis_img[:,:,0]+val)>255
    if val< 0:
        mask = (dis_img[:,:,0]+val)<0
    dis_img[:,:,0] += np.where(mask,0,value)
    hei,wid = dis_img.shape[0:2]
    mid = np.random.randint(0,w)
    factor = np.random.uniform(0.6,0.8)
    if np.random.rand() > .5:
        dis_img[:,0:mid,0] += np.where(mask,0,val)

    hei,wid = dis_img.shape
    horizon = 2*h/5
    v_shift = np.random.randint(-h/8,h/8)
    pts1 = np.float32([[0,horizon],[wid,horizon],[0,hei],[wid,hei]])
    pts2 = np.float32([[0,hrozion+v_shift],[wid,horizon+v_shift],[0,hei],[wid,hei]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dis_img = cv2.warpPerspective(new_img,M,(wid,hei),borderMode=cv2.BORDER_REPLICATE)
    return (dis_img.astype(np.uint8),angle)


def generate_training_data(images,angles,batct_size=128,validation_flag=False):
    """
    Method for model training data generator to load,process and distort images, yield them to model
    """
    images,angles = shuffle(images,angles)
    X,y = ([],[])
    while True:
        for i in range(len(angles)):
            img = cv2.imread(images[i])
            angle = angles[i]
            img = preprocess_image(img)
            if not validation_flag:
                img,angle = distort_img(img,angle)
            X.append(img)
            y.append(angle)

            if len(X) == batch_size:
                yield (np.array(X),np.array(y))
                X,y = ([],[])
                images,angles = shuffle(images,angles)

            if abs(angle)>0.33:
                img = cv2.flip(img,1)
                angle *= -1
                X.append(img)
                y.append(angle)
                if len(X) == batch_size:
                    yield (np.array(X), np.array(y))
                    X,y = ([],[])
                    imges,angles = shuffle(images,angles)


def generate_visualtraining(images,angles,batch_size=20,validation_flag=False):
    """
    Method for loading, procesisng ad distorting images
    """
    imges,angles = shuffle(images,angles)
    for i in range(batch_size):
        img = cv2.imread(images)
        angle = angles[i]
        img = preprocess_image(img)
        if not validation_flag:
            img,angle = distort_img(img,angle)
        X.append(img)
        y.append(angle)
    return (np.array(X),np.array(y))

#
#MAIN CODE STARTS HERE
#
source_path = os.getcwd() + "../data/IMG"
csv_path = os.getcwd() + "../data/driving_data_log.csv"

images = []
angles = []

with open(csv_path,'rt') as csv:
    driving_data = list(csv.reader(csv, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE))

for row in driving_data[1:]:
    if float(row[6])<0.1:
        continue
    #Center image and angle
    images.append(source_path + row[0])
    angles.append(float(row[3]))
    #Left image and angle
    images.append(source_path + row[1])
    angles.append(float(row[3] + 0.25))
    #Right image and angle
    images.append(source_path + row[2])
    angles.append(float(row[3]))

images = np.array(images)
angles = np.array(angles)

#Print a histogram to represent angles and number of images
print('Before:', images.shape, angles.shape)
num_bins = 23
avg_samples = len(angles)/num_bins
hist,bins = np.histogram(angles, num_bins)
widht = 0.7 * (bins[1] - bins[0])
center = (bins[:1] + bins[1:])/2
plt.bar(center, hist, align='center', width=width)
plt.plot((np.min(angles), np.max(angles)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
plt.show()

#Keep-prob for each bin
keep_probs = []
target = avg_samples_per_bin * .5
for i in range(num_bins):
    if hist[i] < target:
        keep_probs.append(1.)
    else:
        keep_probs.append(1./(hist[i]/target))
remove_list = []

for i in range(len(angles)):
    for j in range(num_bins):
        if angles[i] > bins[j] and angles[i] <= bins[j+1]:
            if np.random.rand() > keep_probs[j]:
                remove_list.append(i)
images = np.delete(images, remove_list, axis=0)
angles = np.delete(angles, remove_list)

# print histogram again to show more even distribution of steering angles
hist, bins = np.histogram(angles, num_bins)
plt.bar(center, hist, align='center', width=width)
plt.plot((np.min(angles), np.max(angles)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
#plt.show()

print('After:', images.shape, angles.shape)

# visualize a single batch of the data
X,y = generate_training_data_for_visualization(images, angles)
visualize_dataset(X,y)

# split into train/test sets
images_train, images_test, angles_train, angles_test = train_test_split(images, angles,test_size=0.05, random_state=42)
print('Train:', images_train.shape, angles_train.shape)
print('Test:', images_test.shape, angles_test.shape)

#
#Convolution Neural Network
#

just_checkin_data = False

if not just_checkin_data:
    model = Sequential()

    #Noramlize data
    model.add(Lambda(lambda x:x/127.5 - 1.0, input_shape=(66,200,3)))

    #Adding three 5x5 Convolution layers
    #Output Depth 24,36,48
    #Stride 2x2
    model.add(Convolution2D(24,5,5,subsample=(2,2), border_mode='valid', W_regularizer=12(0.001)))
    model.add(ELU())
    model.add(Convolution2D(36,5,5,subsample=(2,2), border_mode='valid', W_regularizer=12(0.001)))
    model.add(ELU())
    model.add(Convolution2D(48,5,5,subsample=(2,2), border_mode='valid', W_regularizer=12(0.001)))
    model.add(ELU())

    #Adding two 3x3 layers
    #Output depth 64 & 64
    model.add(Convolution2D(64,3,3, border_mode='valid', W_regularizer=12(0.001)))
    model.add(ELU())
    model.add(Convolution2D(64,3,3, border_mode='valid', W_regularizer=12(0.001)))
    model.add(ELU())

    #Adding flatten Layer
    model.add(Flatten())

    #Adding three fully connected layers
    #Output depth 100,50,10
    #Activation: tanh
    model.add(Debse(100, W_regularizer=12(0.001)))
    mode.add(ELU())
    model.add(Debse(50, W_regularizer=12(0.001)))
    mode.add(ELU())
    model.add(Debse(10, W_regularizer=12(0.001)))
    mode.add(ELU())

    #Adding fully connected output layer
    model.add(Dense(1))

    #Compile and train
    model.compile(optimizer=Adma(1r=1e-4),loss='mse')

    #initialize generators
    train_gen = generate_training_data(imges_train,angles_train,validation_falg=False,batch_size=64)

    valid_gen = generate_training_data(imges_train,angles_train,validation_falg=True,batch_size=64)


    tes_gen = generate_training_data(imges_test,angles_test,validation_falg=False,batch_size=64)

    checkpoint = ModelCheckpoint('model{epoch:02d}.h5')

    history = model.fit_generator(train_gen,validation_data=valid_gen,nb_val_samples=2560,samples_per_epoch=23040, nb_epoch=5,verbose=2, callbacks=[checkpoint])

    print(model.summary())

    n=12
    X_test,y_test = generate_visualtraining(images_test[:,n], angles_test[:n], batch_size=n,validation_flag=True)
    y_pred = model.predict(X_test,y_test,y_pred)
    visualize_dataset(X_test, y_test, y_pred)

    #Save model data
    model.save_weights('./model.h5')
    json_string - model.to_json()
    with open('./model.json','w')  as f:
        f.write(json_string)
