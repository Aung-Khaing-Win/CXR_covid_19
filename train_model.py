# USAGE
# python train.py --dataset dataset

# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import warnings
import argparse
import sys
import cv2
import os

if not sys.warnoptions:
    warnings.simplefilter('ignore')


sns.set_style('whitegrid')

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default='train',
    help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
    help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str, default="model.model",
    help="path to output loss/accuracy plot")
ap.add_argument('-a', '--alpha', type=float, default=0.001,
                help='Learning Rate')
ap.add_argument('-e', '--epochs', type=int, default=25,
                help='Epochs')
ap.add_argument('-b', '--batchs', type=int, default=4,
                help='Batchs')
ap.add_argument('-w', '--weights', type=str, default='imagenet',
                help='VGG16 - weights')
ap.add_argument('-l1', '--l1_activation', type=str, default='relu',
                help='Layer 1 Activation model')
ap.add_argument('-l2', '--l2_activation', type=str, default='softmax',
                help='Layer 2 Activation model')
ap.add_argument('-ip', '--info_path', type=str, default='model_info',
                help='Layer 2 Activation model')
ap.add_argument('-v', '--validation', type=str, default='y',
                help='Model validation on hold out dataset')
ap.add_argument('-vp', '--validation_path', type=str, default='validation',
                help='Path to validation dataset')

args = vars(ap.parse_args())

# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = args['alpha']
EPOCHS = args['epochs']
BS = args['batchs']
VGG_WEIGHTS = args['weights']
L1_ACTIVATION = args['l1_activation']
L2_ACTIVATION = args['l2_activation']
MODEL_INFO_PATH = args['info_path']
PLOT_PATH = os.path.join(MODEL_INFO_PATH, args['plot'])
VALIDATION_PATH = args['validation_path']

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images

print('\n\n\n')
print('*****************************************************************')
print('Chest X-Ray Clasification')
print('*****************************************************************')
print()
print()
print('-----------------------------------------')
print('Model Parameters')
print('-----------------------------------------')
print('model:\t\t\t\t', 'VGG16')
print('model-weight:\t\t\t', VGG_WEIGHTS)
print('layer 1 activation function:\t', L1_ACTIVATION)
print('layer 2 activation function:\t', L2_ACTIVATION)
print('alpha:\t\t\t\t', INIT_LR)
print('epochs:\t\t\t\t', EPOCHS)
print('batchs:\t\t\t\t', BS)

model_parameters = {
    'model': 'VGG16',
    'vgg-weights': VGG_WEIGHTS,
    'layer_1_activation': L1_ACTIVATION,
    'layer_2_activation': L2_ACTIVATION,
    'alpha':INIT_LR,
    'epochs': EPOCHS,
    'batchs': BS
}

f = open(os.path.join(MODEL_INFO_PATH, 'parameters.txt'), 'w')
print(model_parameters, file = f)

print()
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# loop over the image paths
for imagePath in imagePaths:
    # extract the class label from the filename
    label = imagePath.split(os.path.sep)[-2]

    # load the image, swap color channels, and resize it to be a fixed
    # 224x224 pixels while ignoring aspect ratio
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    # update the data and labels lists, respectively
    data.append(image)
    labels.append(label)

# convert the data and labels to NumPy arrays while scaling the pixel
# intensities to the range [0, 255]
data = np.array(data) / 255.0
labels = np.array(labels)


# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Save Labels
f = open(os.path.join(MODEL_INFO_PATH + '_labels.txt'), 'w')
print(lb.classes_, file = f)

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
    test_size=0.20, stratify=labels, random_state=42)

# initialize the training data augmentation object
trainAug = ImageDataGenerator(
    rotation_range=15,
    fill_mode="nearest")

# load the VGG16 network, ensuring the head FC layer sets are left
# off
baseModel = VGG16(weights=VGG_WEIGHTS, include_top=False,
    input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation=L1_ACTIVATION)(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation=L2_ACTIVATION)(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
    layer.trainable = False

# compile our model
print()
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
    metrics=["accuracy"])

# train the head of the network
print()
print("[INFO] training head...")
H = model.fit_generator(
    trainAug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)

# plot the training loss and accuracy
N = EPOCHS
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="Train Loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="Validation Loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="Train Accuracy")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="Validation Accuracy")
plt.title("Comparing Accuracy vs Loss on CXR Classification Model")
plt.xlabel("No. of Epochs")
plt.ylabel("Accuracy & Loss")
ax = plt.gca()
ax.yaxis.grid(linestyle=':', linewidth=1)
ax.xaxis.grid(False)
for spine in ['left', 'top', 'right', 'bottom']:
    ax.spines[spine].set_visible(False)

plt.legend(loc="center right", prop={'size': '7'})
plt.savefig(PLOT_PATH)

# Save history
df_history = pd.DataFrame.from_dict(H.history, orient='columns')
df_history.to_csv(os.path.join(MODEL_INFO_PATH,'trained_history.csv'),
                  index=True, encoding='UTF-8')

def evaluate_model(model, data, label, dataset_name, lb=lb):
    print()
    print('[INFO] Evaluating network for', dataset_name)
    # Make predictions
    model_predictions = model.predict(data, batch_size=BS)
    model_predictions = np.argmax(model_predictions, axis=1)

    # Calculate precision, recall, f1-score and support

    report = classification_report(np.argmax(label, axis=1), model_predictions, target_names=lb.classes_)
    prfs = precision_recall_fscore_support(np.argmax(label, axis=1), model_predictions)
    print(prfs)
    prfs = {
        'precision': np.round(prfs[0], 2),
        'recall': np.round(prfs[1], 2),
        'f1-score': np.round(prfs[2], 2),
        'support': np.round(prfs[3])
    }

    df_prfs = pd.DataFrame.from_dict(prfs, orient='index', columns=lb.classes_)
    df_prfs.to_csv(os.path.join(MODEL_INFO_PATH, dataset_name + '_precision_recall_f1.csv'), encoding='UTF-8')

    cm = confusion_matrix(label.argmax(axis=1), model_predictions)
    total = sum(sum(cm))
    acc = (cm[0, 0] + cm[1, 1]) / total
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

    model_score = {
        'accuracy': acc,
        'sensitivity': sensitivity,
        'specificity': specificity
    }

    model_score = pd.DataFrame.from_dict(model_score, orient='index')
    model_score.to_csv(os.path.join(MODEL_INFO_PATH, dataset_name + '_model_score.csv'), encoding=False)
    # show the confusion matrix, accuracy, sensitivity, and specificity
    print()
    print('Model Score -', dataset_name)
    print('..............................................................')
    print("accuracy:\t {:.4f}".format(acc))
    print("sensitivity:\t {:.4f}".format(sensitivity))
    print("specificity:\t {:.4f}".format(specificity))
    print()
    print('Classification Report -', dataset_name)
    print('..............................................................')
    f = open(os.path.join(MODEL_INFO_PATH, dataset_name + '_console_outpout.txt'), 'w')
    print(report, file = f)
    print(report)


print('-----------------------------------------')
print('Model Hyper Parameters')
print('-----------------------------------------')
print('model:\t\t\t\t', 'VGG16')
print('model-weight:\t\t\t', VGG_WEIGHTS)
print('layer 1 activation function:\t', L1_ACTIVATION)
print('layer 2 activation function:\t', L2_ACTIVATION)
print('alpha:\t\t\t\t', INIT_LR)
print('epochs:\t\t\t\t', EPOCHS)
print('batchs:\t\t\t\t', BS)

evaluate_model(model, testX, testY, dataset_name='test_dataset', lb=lb)
evaluate_model(model, trainX, trainY, dataset_name='train_dataset', lb=lb)

if args['validation'].lower() in ['y', 'yes']:
    images = list(paths.list_images(VALIDATION_PATH))
    data, labels = [], []
    for path in images:
        label = path.split(os.path.sep)[1]

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))

        data.append(image)
        labels.append(label)
    data = np.array(data) / 255.0
    labels = np.array(labels)

    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)
    evaluate_model(model, data, labels, dataset_name='validation_dataset', lb=lb)

# serialize the model to disk
print("[INFO] saving model...")
model.save(args["model"], save_format="h5")

print("[INFO] Done.")
print()
print()
print('This model has been adapted from the following article.')
print()
print('Article   : Detecting COVID-19 in X-ray images with Keras, TensorFlow, and Deep Learning ')
print('Author    : Adrian Rosebrock')
print('Posted on : March 16, 2020')
print('Link      : https://www.pyimagesearch.com/2020/03/16/detecting-covid-19-in-x-ray-images-with-keras-tensorflow-and-deep-learning/')
print()




