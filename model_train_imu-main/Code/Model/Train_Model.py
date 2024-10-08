import pandas as pd 
import tensorflow as tf 
import glob
import numpy as np 
from tensorflow import keras
from keras import Sequential
from keras.callbacks import EarlyStopping
from sklearn.utils import shuffle
import sklearn.model_selection
import matplotlib.pyplot as plt

# Define early stopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)



SAMPLES_PER_LABEL = 60
SENSOR_NUM = 6
NUM_CLASSESS = 2

# Dictionary to convert filenames to labels
GESTURES = {'Breathing': 0, 'SBreathing': 1}

# Function to normalize a column to the range [-1, 1]
def normalize_column(column):
    return 2 * (column - column.min()) / (column.max() - column.min()) - 1

# Function to read files and append labels based on filename
def load_data_from_label_dict(label_dict):
    data_list = []
    for label_name, label_value in label_dict.items():
        file_path_pattern = f'./data_train/{label_name}.csv'
        for file_path in glob.glob(file_path_pattern):
            df = pd.read_csv(file_path)
            df['label'] = label_value
            data_list.append(df)
    return pd.concat(data_list, ignore_index=True)

# Load data
df = load_data_from_label_dict(GESTURES)

# Normalize the dataset columns
df[["aX", "aY", "aZ", "gX", "gY", "gZ"]] = df[["aX", "aY", "aZ", "gX", "gY", "gZ"]].apply(normalize_column)

# Create the dataset after normalization
dataSet = df[["aX", "aY", "aZ", "gX", "gY", "gZ", "label"]]

# print(dataSet)

x = np.array(dataSet.drop("label", axis=1))
y = np.array(dataSet["label"])

modDataset = []
modTruth =[]

for i in range(len(x)-SAMPLES_PER_LABEL):
    temp = []
    for j in range(i, i+SAMPLES_PER_LABEL):
        temp.append(x[j])
    modDataset.append(temp)

for i in range(len(y)-SAMPLES_PER_LABEL):
    temp = []
    for j in range(i, i+SAMPLES_PER_LABEL):
        temp.append(y[j])
    
    most_common_item = max(temp, key = temp.count)

    modTruth.append(most_common_item)

print(len(modDataset))
print(len(modTruth))

print(len(modDataset[0]))
print(modDataset[1])

modDataset = np.array(modDataset).reshape(-1, SAMPLES_PER_LABEL, SENSOR_NUM)


y = np.array(modTruth)
x = modDataset

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size = 0.2)

print(x_train)
print(y_train)

model = Sequential()
model.add(keras.layers.Flatten(input_shape=(SAMPLES_PER_LABEL, SENSOR_NUM)))
# model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(16, activation='sigmoid'))
model.add(keras.layers.Dense(16, activation='sigmoid'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(NUM_CLASSESS, activation='sigmoid'))   

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]) 
model.summary()

history = model.fit(x_train,y_train, epochs=100, validation_data=(x_test, y_test), callbacks=[early_stop])


# Convert the model to the TensorFlow Lite format without quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# model.save('gesture_model.tflite')
with open('AImodel.tflite', 'wb') as f:
    f.write(tflite_model)

# Vẽ biểu đồ Loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Vẽ biểu đồ Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

pred = model.predict(x_test)
results = np.argmax(pred, axis=1)


def python_to_c_array(array_name, data, columns=4):
    """
    Convert a 1D Python array to a C array initialization code with specified number of columns.
    
    Args:
        array_name (str): Name of the C array variable.
        data (list): List containing the data elements.
        columns (int): Number of columns to use in the C array initialization code.
    
    Returns:
        str: C code for initializing the array.
    """
    c_code = f"float {array_name}[{len(data)}] = {{\n"
    for i in range(0, len(data), columns):
        row_data = ", ".join([f"{val:.3f}" for val in data[i:i+columns]])
        c_code += f"    {row_data},\n"
    c_code += "};\n"
    return c_code


# Convert the model to the TensorFlow Lite format without quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

import os
basic_model_size = os.path.getsize("AImodel.tflite")
print("Model is %d bytes" % basic_model_size)


# Function: Convert some hex value into an array for C programming
def hex_to_c_array(hex_data, var_name):

  c_str = ''

  # Create header guard
  c_str += '#ifndef ' + var_name.upper() + '_H\n'
  c_str += '#define ' + var_name.upper() + '_H\n\n'

  # Add array length at top of file
  c_str += '\nunsigned int ' + var_name + '_len = ' + str(len(hex_data)) + ';\n'

  # Declare C variable
  c_str += 'unsigned char ' + var_name + '[] = {'
  hex_array = []
  for i, val in enumerate(hex_data) :

    # Construct string from hex
    hex_str = format(val, '#04x')

    # Add formatting so each line stays within 80 characters
    if (i + 1) < len(hex_data):
      hex_str += ','
    if (i + 1) % 12 == 0:
      hex_str += '\n '
    hex_array.append(hex_str)

  # Add closing brace
  c_str += '\n ' + format(' '.join(hex_array)) + '\n};\n\n'

  # Close out header guard
  c_str += '#endif //' + var_name.upper() + '_H'

  return c_str

# Write TFLite model to a C source (or header) file
c_model_name = 'AImodel'
with open(c_model_name + '.h', 'w') as file:
  file.write(hex_to_c_array(tflite_model, c_model_name))
