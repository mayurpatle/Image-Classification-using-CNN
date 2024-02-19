Overall Flow:

Load and preprocess image data into training and validation sets
Build the CNN model
Train the CNN on training data
Evaluate on validation data
Make predictions on new images
Key Components:

Convolutional Layers:

The convolutional layers are the core building blocks of a CNN. They apply a convolution operation to the input using a set of learnable filters to extract features.
The filters slide over the input image and detect visual patterns like edges, colors, textures etc. Multiple filters are used to detect multiple features.
The convolution layers help pick up on local spatial patterns across the image.
Pooling Layers:

Pooling layers downsamples the image resolution to reduce computations. Spatial dimensionality reduction also helps create a more robust representation of key features.
There are different types of pooling like max, average etc. Here max pooling takes the maximum pixel value in each filter region.
Flatten Layer:

Flatten converts the final convolutional layer output to a 1D vector. This is done to feed the features into a fully connected neural network.
Dense Layers:

These are regular fully connected neural network layers. They interpret the features extracted by the convolutional layers for the final classification.
Multiple dense layers are used to learn hierarchical feature representations of the image.
The model uses binary crossentropy loss for binary classification and backpropagation to train the above components end-to-end to minimize the loss.

# Importing the libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
tf.__version__

    This section imports the necessary libraries. TensorFlow is used for building and training the neural network, and ImageDataGenerator from Keras is used for data augmentation during preprocessing.

python

# Part 1 - Data Preprocessing

# Preprocessing the Training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

    Data preprocessing for the training set is performed here. The ImageDataGenerator is used to perform data augmentation on the training images. It rescales the pixel values, applies shear transformations, zooms in, and flips horizontally. The flow_from_directory method is used to load and preprocess the images from the 'dataset/training_set' directory.

python

# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

    Similar to the training set, the test set is preprocessed using the ImageDataGenerator. However, no additional augmentation is applied to the test set.

python

# Part 2 - Building the CNN

# Initialising the CNN
cnn = tf.keras.models.Sequential()

    A sequential model is initialized. This allows the neural network to be built layer by layer in a step-by-step fashion.

python

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

    The first convolutional layer is added to the model. It consists of 32 filters, each with a 3x3 kernel, using the ReLU activation function. The input shape is set to (64, 64, 3), indicating the size and number of color channels in the input images.

python

# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    A max-pooling layer is added to reduce the spatial dimensions of the convolved features by taking the maximum value in a 2x2 window with a stride of 2.

python

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    A second convolutional layer with max-pooling is added. This helps the model to learn more complex features from the input images.

python

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

    The flatten layer is added to convert the 2D feature maps into a 1D vector, preparing the data for the fully connected layers.

python

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

    A fully connected layer with 128 neurons and ReLU activation is added.

python

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    The output layer with a single neuron and sigmoid activation function is added for binary classification (cat or dog).

python

# Part 3 - Training the CNN

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    The model is compiled with the Adam optimizer, binary crossentropy loss function (suitable for binary classification), and accuracy as the metric to monitor during training.

python

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)

    The CNN is trained on the training set using the fit method. The training is performed for 25 epochs, and the model's performance is evaluated on the test set.

python

# Part 4 - Making a single prediction

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)

    A single prediction is made on a new image ('dataset/single_prediction/cat_or_dog_1.jpg'). The image is loaded, converted to a NumPy array, and expanded to match the expected input shape. The model predicts whether it's a cat or a dog, and the result is printed. The training_set.class_indices helps map the prediction value to the corresponding class label. If the predicted value is close to 1, it is classified as a dog; otherwise, it's classified as a cat.

