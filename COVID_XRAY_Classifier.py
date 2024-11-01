#Data Preprocessing
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from glob import glob
from PIL import Image
import os
import random
import cv2

#Model
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model,load_model
from tensorflow.keras.layers import Activation,Dense, Dropout, Flatten, Conv2D, MaxPooling2D,MaxPool2D,AveragePooling2D,GlobalMaxPooling2D
from tensorflow.keras import backend as K
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical # convert to one-hot-encoding
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator,array_to_img
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping,ModelCheckpoint
from tensorflow.keras.metrics import PrecisionAtRecall,Recall

tf.config.list_physical_devices("GPU")

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import os

# Define the output folder path
output_folder = "/Users/lovepreetsinghgill/COVID-19_XRAY_Classifier/results"

# Create the folder if it doesn't already exist
os.makedirs(output_folder, exist_ok=True)

from numpy.random import seed
seed(0)

tf.random.set_seed(0)

path = '/Users/lovepreetsinghgill/COVID-19_XRAY_Classifier/data/COVID-19_Radiography_Dataset'
image_files = glob(os.path.join(path, '*', '*', '*.png'))
print(f"Number of images found: {len(image_files)}")


diag_code_dict = {
    'COVID': 0,
    'Lung_Opacity': 1,
    'Normal': 2,
    'Viral Pneumonia': 3}

diag_title_dict = {
    'COVID': 'Covid-19',
    'Lung_Opacity': 'Lung Opacity',
    'Normal': 'Healthy',
    'Viral Pneumonia': 'Viral Pneumonia'}
imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(path, '*', '*', '*.png'))}
imageid_path_dict
covidData = pd.DataFrame.from_dict(imageid_path_dict, orient='index').reset_index()
covidData.columns = ['image_id', 'path']

#imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(path, '*','*.png'))}

#imageid_path_dict

#covidData = pd.DataFrame.from_dict(imageid_path_dict, orient = 'index').reset_index()
#covidData.columns = ['image_id','path']
classes = covidData.image_id.str.split('-').str[0]
covidData['diag'] = classes
covidData['target'] = covidData['diag'].map(diag_code_dict.get)
covidData['Class'] = covidData['diag'].map(diag_title_dict.get)

samples,feature = covidData.shape
duplicated = covidData.duplicated().sum()
null_values = covidData.isnull().sum().sum()

print('Simple EDA')
print('Number of samples: %d'%(samples))
print('duplicates: %d'%(duplicated))
print('null values: %d' %(null_values))

#samples per class
plt.figure(figsize=(20,8))
sns.set(style="ticks", font_scale = 1)
ax = sns.countplot(data = covidData,x='Class',order = covidData['Class'].value_counts().index,palette="flare")
sns.despine(top=True, right=True, left=True, bottom=False)
plt.xticks(rotation=0,fontsize = 12)
ax.set_xlabel('Sample Type - Diagnosis',fontsize = 14,weight = 'bold')
ax.set(yticklabels=[])
ax.axes.get_yaxis().set_visible(False) 
plt.title('Number of Samples per Class', fontsize = 16,weight = 'bold');
plt.savefig(os.path.join(output_folder, "sample_count_plot.png"))
plt.close()

#plot percentage
for p in ax.patches:
    ax.annotate("%.1f%%" % (100*float(p.get_height()/samples)), (p.get_x() + p.get_width() / 2., abs(p.get_height())),
    ha='center', va='bottom', color='black', xytext=(0, 10),rotation = 'horizontal',
    textcoords='offset points')

covidData['image'] = covidData['path'].map(lambda x: np.asarray(Image.open(x).resize((75, 75))))

n_samples =3

fig, m_axs = plt.subplots(4, n_samples, figsize = (4*n_samples, 3*4))

for n_axs, (type_name, type_rows) in zip(m_axs,covidData.sort_values(['diag']).groupby('diag')):
    n_axs[1].set_title(type_name,fontsize = 14,weight = 'bold')
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=1234).iterrows()):       
        picture = c_row['path']
        image = cv2.imread(picture)
        c_ax.imshow(image)
        c_ax.axis('off')

plt.savefig(os.path.join(output_folder, "data.png"))
plt.close()

plt.figure()
pic_id = random.randrange(0, samples)
picture = covidData['path'][pic_id]
image - cv2.imread(picture)
plt.imshow(image)
plt.axis('off')
plt.savefig(os.path.join(output_folder, "lung.png"))
plt.close()

print('shape of the image: {}'.format(image.shape))

print('image size {}'.format(image.size))

image.dtype

print('max rgb: {}'.format(image.max()))

print('min rgb: {}'.format(image.min()))

image[0, 0]

plt.title('Bchannel', fontsize=14, weight='bold')
plt.imshow(image[:,:,0])
plt.axis('off');
plt.savefig(os.path.join(output_folder, "Bchannel.png"))
plt.close()

mean_val = []
std_dev_val = []
max_val = []
min_val = []

for i in range(0, samples):
    mean_val.append(covidData['image'][i].mean())
    std_dev_val.append(np.std(covidData['image'][i]))
    max_val.append(covidData['image'][i].max())
    min_val.append(covidData['image'][i].min())

imageEDA = covidData.loc[:, ['image', 'Class', 'path']]
imageEDA['mean'] = mean_val
imageEDA['stedev'] = std_dev_val
imageEDA['max'] = max_val
imageEDA['min'] = min_val

subt_mean_samples = imageEDA['mean'].mean() - imageEDA['mean']
imageEDA['subt_mean'] = subt_mean_samples

ax = sns.displot(data = imageEDA, x = 'mean', kind="kde")
plt.title('Image color mean value distribution')

ax = sns.displot(data = imageEDA, x = 'mean', kind="kde", hue='Class')
plt.title('Image color mean value distribution by class')

ax = sns.displot(data = imageEDA, x = 'max', kind="kde", hue='Class')
plt.title('Image color max value distribution by class')

sns.displot(data = imageEDA, x = 'min', kind="kde", hue='Class')
plt.title('Image color min value distribution by class')

plt.figure(figsize=(20,8))
sns.set(style="ticks", font_scale = 1)
ax = sns.scatterplot(data=imageEDA, x="mean", y=imageEDA['stedev'], hue = 'Class',alpha=0.8);
sns.despine(top=True, right=True, left=False, bottom=False)
plt.xticks(rotation=0,fontsize = 12)
ax.set_xlabel('Image Channel Colour Mean',fontsize = 14,weight = 'bold')
ax.set_ylabel('Image Channel Colour Standard Deviation',fontsize = 14,weight = 'bold')
plt.title('Mean and Standard Deviation of Image Samples', fontsize = 16,weight = 'bold');

plt.figure(figsize=(20,8));
g = sns.FacetGrid(imageEDA, col="Class", height=5)
g.map_dataframe(sns.scatterplot, x='mean', y='stedev');
g.set_titles(col_template="{col_name}", row_template="{row_name}", size=16)
g.fig.subplots_adjust(top=.7)
g.fig.suptitle('Mean anfd standard dev of img samples')
axes = g.axes.flatten()
axes[0].set_ylabel('std dev')
for ax in axes:
    ax.set_xlabel('Mean')
g.fig.tight_layout()

path = '/Users/lovepreetsinghgill/COVID-19_XRAY_Classifier/data/COVID-19_Radiography_Dataset'

classes = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
num_classes = len(classes)
batch_size = 16

train_datagen = ImageDataGenerator(rescale=1./255,
                                  rotation_range=20,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  horizontal_flip=True, validation_split=0.2)

test_datagen = ImageDataGenerator(rescale=1./255,
                                 validation_split=0.2)

train_gen = train_datagen.flow_from_directory(directory=path,
                                             target_size=(299, 299),
                                             class_mode='categorical',
                                             subset='training',
                                             shuffle=True, classes=classes,
                                             batch_size=batch_size,
                                             color_mode="grayscale")

test_gen = test_datagen.flow_from_directory(directory=path,
                                             target_size=(299, 299),
                                             class_mode='categorical',
                                             subset='validation',
                                             shuffle=False, classes=classes,
                                             batch_size=batch_size,
                                             color_mode="grayscale")

model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', padding='Same', input_shape=(299, 299, 1)))
model.add(BatchNormalization())

################

model.add(Conv2D(64, (3,3), activation='relu', padding='Same'))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), activation='relu', padding='Same'))
model.add(BatchNormalization())

model.add(Conv2D(64, (3,3), activation='relu', padding='Same'))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

#################

model.add(Conv2D(64, (3,3), activation='relu', padding='Same'))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), activation='relu', padding='Same'))
model.add(BatchNormalization())

model.add(Conv2D(64, (3,3), activation='relu', padding='Same'))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

#################

model.add(Flatten())

model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))

#Output 
model.add(BatchNormalization())
model.add(Dense(num_classes, activation='softmax'))

model.summary()

opt = Adam(learning_rate=0.002)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy',Recall()])

#Model params
epochs=300

#callback
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5, min_lr=0.00001)
early_stopping_monitor = EarlyStopping(patience=100, monitor='val_loss', mode='min')

# Define the callbacks list
callbacks_list = [learning_rate_reduction, early_stopping_monitor]

batch_x, batch_y = next(train_gen)  # Or use test_gen for validation data
print("Train batch X shape:", batch_x.shape)
print("Train batch Y shape:", batch_y.shape)


history = model.fit(
    train_gen,
    steps_per_epoch=len(train_gen) // batch_size,
    validation_steps=len(test_gen) // batch_size,
    validation_data=test_gen,
    epochs=epochs,
    callbacks=callbacks_list  # Pass callbacks_list directly
)


model.save(os.path.join(output_folder, "covid_classifier_model.h5"))
history_df = pd.DataFrame(history.history)
history_df.to_csv(os.path.join(output_folder, "training_history.csv"), index=False)

y_pred = model.predict(test_gen)
history.history

fig, axarr = plt.subplots(1,3, figsize=(15,5), sharex=True)

sns.set(style="ticks", font_scale = 1)
sns.despine(top=True, right=True, left=False, bottom=False)

historyDF = pd.DataFrame.from_dict(history.history)
ax = sns.lineplot(x = historyDF.index, y = history.history['recall'], ax=axarr[0], label="Training");
ax = sns.lineplot(x = historyDF.index, y = history.history['val_recall'], ax=axarr[0], label="Validation");
ax.set_ylabel('Recall')

ax = sns.lineplot(x = historyDF.index, y = history.history['loss'], ax=axarr[1], label="Training");
ax = sns.lineplot(x = historyDF.index, y = history.history['val_loss'], ax=axarr[1], label="Validation");
ax.set_ylabel('Loss')

ax = sns.lineplot(x = historyDF.index, y = history.history['lr'], ax=axarr[2]);
ax.set_ylabel('Learning Rate')

axarr[0].set_title('Training and Validation metric recall')
axarr[1].set_title('Training and Validation loss')
axarr[2].set_title('lr during training')

for ax in axarr:
    ax.set_xlabel('Epochs')

plt.suptitle('Training and performance plots', fontsize=16, weight='bold');
fig.tight_layout(pad=3.0)
plt.savefig(os.path.join(output_folder, "Training_and_performance_plots.png"))
plt.close()()

predictions = np.array(list(map(lambda x: np.argmax(x), y_pred)))

y_true=test_gen.classes

CMatrix = pd.DataFrame(confusion_matrix(y_true, predictions), columns=classes, index =classes)

plt.figure(figsize=(12, 6))
ax = sns.heatmap(CMatrix, annot = True, fmt = 'g' ,vmin = 0, vmax = 250,cmap = 'Blues')
ax.set_xlabel('Predicted',fontsize = 14,weight = 'bold')
ax.set_xticklabels(ax.get_xticklabels(),rotation =0);

ax.set_ylabel('Actual',fontsize = 14,weight = 'bold') 
ax.set_yticklabels(ax.get_yticklabels(),rotation =0);
ax.set_title('Confusion Matrix - Test Set',fontsize = 16,weight = 'bold',pad=20);

plt.savefig(os.path.join(output_folder, "confusion_matrix.png"))
plt.close()

acc = accuracy_score(y_true, predictions)

results_all = precision_recall_fscore_support(y_true, predictions, average='macro', zero_division=1)
results_class = precision_recall_fscore_support(y_true, predictions, average=None, zero_division=1)

metric_columns = ['Precision','Recall', 'F-Score','S']
all_df = pd.concat([pd.DataFrame(list(results_class)).T,pd.DataFrame(list(results_all)).T])
all_df.columns = metric_columns
all_df.index = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia','Total']

def metrics_plot(df,metric):
    plt.figure(figsize=(22,10))
    ax = sns.barplot(data =df, x=df.index, y = metric,palette = "Blues_d")
    #Bar Labels
    for p in ax.patches:
        ax.annotate("%.1f%%" % (100*p.get_height()), (p.get_x() + p.get_width() / 2., abs(p.get_height())),
        ha='center', va='bottom', color='black', xytext=(-3, 5),rotation = 'horizontal',textcoords='offset points')
    sns.despine(top=True, right=True, left=True, bottom=False)
    ax.set_xlabel('Class',fontsize = 14,weight = 'bold')
    ax.set_ylabel(metric,fontsize = 14,weight = 'bold')
    ax.set(yticklabels=[])
    ax.axes.get_yaxis().set_visible(False) 
    plt.title(metric+ ' Results per Class', fontsize = 16,weight = 'bold');

metrics_plot(all_df, 'Precision')#Results by Class
metrics_plot(all_df, 'Recall')#Results by Class
metrics_plot(all_df, 'F-Score')#Results by Class
print('**Overall Results**')
print('Accuracy Result: %.2f%%'%(acc*100)) #Accuracy of the whole Dataset
print('Precision Result: %.2f%%'%(all_df.iloc[4,0]*100))#Precision of the whole Dataset
print('Recall Result: %.2f%%'%(all_df.iloc[4,1]*100))#Recall of the whole Dataset
print('F-Score Result: %.2f%%'%(all_df.iloc[4,2]*100))#F-Score of the whole Dataset

with open(os.path.join(output_folder, "metrics_summary.txt"), "w") as f:
    f.write(f"Accuracy: {acc:.2f}\n")
    f.write(f"Precision: {all_df.iloc[4,0]*100:.2f}%\n")
    f.write(f"Recall: {all_df.iloc[4,1]*100:.2f}%\n")
    f.write(f"F-Score: {all_df.iloc[4,2]*100:.2f}%\n")

