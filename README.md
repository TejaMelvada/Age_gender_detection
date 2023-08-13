import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

# Load and preprocess data
data_generator = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)
batch_size = 32

train_data = data_generator.flow_from_directory(
    'path/to/training_data',
    target_size=(224, 224),
    batch_size=batch_size,
    subset='training',
    class_mode='categorical'
)

valid_data = data_generator.flow_from_directory(
    'path/to/training_data',
    target_size=(224, 224),
    batch_size=batch_size,
    subset='validation',
    class_mode='categorical'
)

# Build and train the model
base_model = MobileNetV2(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
gender_output = Dense(2, activation='softmax', name='gender_output')(x)
age_output = Dense(num_age_groups, activation='softmax', name='age_output')(x)

model = Model(inputs=base_model.input, outputs=[gender_output, age_output])

model.compile(optimizer='adam',
              loss={'gender_output': 'categorical_crossentropy', 'age_output': 'categorical_crossentropy'},
              metrics=['accuracy'])

model.fit(train_data, epochs=10, validation_data=valid_data)

# Make predictions
test_image = tf.keras.preprocessing.image.load_img('path/to/test_image.jpg', target_size=(224, 224))
test_image = tf.keras.preprocessing.image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image = test_image / 255.0

gender_pred, age_pred = model.predict(test_image)

# Convert predictions to actual labels
gender_labels = ['Male', 'Female']
age_group_labels = ['Child', 'Teen', 'Young Adult', 'Adult', 'Senior']
predicted_gender = gender_labels[np.argmax(gender_pred)]
predicted_age_group = age_group_labels[np.argmax(age_pred)]

print("Predicted Gender:", predicted_gender)
print("Predicted Age Group:", predicted_age_group)

