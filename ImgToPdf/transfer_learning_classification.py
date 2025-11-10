# Weight is calculated for synthetic & non synthetic data
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# CONFIGURATION
# -------------------------
dataset_dir = r"D:\RaviKaGroup\ImageProcessing\try1_2110\OutputProcessed"
img_size = (224, 224)
batch_size = 32
epochs_initial = 10
epochs_finetune = 20
learning_rate = 1e-4
finetune_lr = 1e-5

# -------------------------
# DATA LOADING AND SPLIT
# -------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# -------------------------
# CLASS WEIGHTS (to fix imbalance)
# -------------------------
from sklearn.utils.class_weight import compute_class_weight

class_indices = train_generator.class_indices
classes = list(class_indices.keys())
y_train = train_generator.classes
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

print("Class Weights:", class_weights_dict)

# -------------------------
# MODEL BUILDING
# -------------------------
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=img_size + (3,))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
num_classes = len(train_generator.class_indices)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# -------------------------
# CALLBACKS
# -------------------------
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-6)

# -------------------------
# TRAIN TOP LAYERS
# -------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs_initial,
    class_weight=class_weights_dict,
    callbacks=[early_stop, reduce_lr]
)

# -------------------------
# FINE-TUNE LAST BLOCK OF RESNET50
# -------------------------
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=finetune_lr),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs_finetune,
    class_weight=class_weights_dict,
    callbacks=[early_stop, reduce_lr]
)

# -------------------------
# EVALUATION
# -------------------------
val_loss, val_acc = model.evaluate(val_generator)
print(f"✅ Final Validation Accuracy: {val_acc*100:.2f}%")

# -------------------------
# SAVE MODEL
# -------------------------
model.save("resnet50_transfer_balanced.h5")
print("Model saved as 'resnet50_transfer_balanced.h5'")

# -------------------------
# PLOT ACCURACY & LOSS
# -------------------------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'] + history_finetune.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'] + history_finetune.history['val_accuracy'], label='val_acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'] + history_finetune.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'] + history_finetune.history['val_loss'], label='val_loss')
plt.title('Loss')
plt.legend()
plt.show()
