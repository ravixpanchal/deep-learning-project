import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
import random

# -------------------------

# CONFIGURATION

# -------------------------

dataset_dir = r"D:\RaviKaGroup\ImageProcessing\try1_2110\OutputProcessed"
img_size = (224, 224)
batch_size = 32
epochs_initial = 8
epochs_finetune = 15

# -------------------------
# SYNTHETIC AUGMENTATION FOR MINORITY CLASS
# -------------------------

target_class = "output_jpg2"   # folder name for Class 2
target_path = os.path.join(dataset_dir, target_class)
current_images = os.listdir(target_path)
num_current = len(current_images)
target_total = 300  # aim to have ~300 images

if num_current < target_total:
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        brightness_range=(0.7, 1.3),
        fill_mode='nearest'
    )

    i = 0
    print(f"Augmenting {target_class}: {num_current} → {target_total} images")
    while len(os.listdir(target_path)) < target_total:
        img_name = random.choice(current_images)
        img_path = os.path.join(target_path, img_name)
        img = load_img(img_path, target_size=img_size)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)

        for batch in datagen.flow(x, batch_size=1, save_to_dir=target_path, save_prefix='aug', save_format='jpg'):
            i += 1
            if len(os.listdir(target_path)) >= target_total:
                break
    print(f"✅ Augmentation complete: now {len(os.listdir(target_path))} images in {target_class}")
else:
    print(f"{target_class} already has {num_current} images — no augmentation needed.")


# -------------------------

# DATA LOADING

# -------------------------

train_datagen = ImageDataGenerator(
rescale=1./255,
validation_split=0.2,
rotation_range=25,
width_shift_range=0.15,
height_shift_range=0.15,
shear_range=0.15,
zoom_range=0.15,
horizontal_flip=True,
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

num_classes = len(train_generator.class_indices)
print("Classes:", train_generator.class_indices)

# -------------------------

# COMPUTE CLASS WEIGHTS

# -------------------------

class_weights = compute_class_weight(
class_weight='balanced',
classes=np.unique(train_generator.classes),
y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# -------------------------

# MODEL BUILDING

# -------------------------

base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=img_size + (3,))
base_model.trainable = False  # freeze for initial training

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu', kernel_regularizer=l2(1e-4))(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# -------------------------

# LEARNING RATE SCHEDULE

# -------------------------

lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
initial_learning_rate=1e-4,
first_decay_steps=100
)
optimizer = Adam(learning_rate=lr_schedule)

model.compile(optimizer=optimizer,
loss='categorical_crossentropy',
metrics=['accuracy'])

model.summary()

# -------------------------

# INITIAL TRAINING (top layers)

# -------------------------

print("\n---- Initial Training ----")
history = model.fit(
train_generator,
validation_data=val_generator,
epochs=epochs_initial,
class_weight=class_weights
)

# -------------------------

# FINE-TUNE ENTIRE MODEL

# -------------------------

print("\n---- Fine-Tuning Entire Model ----")
base_model.trainable = True  # unfreeze all layers

# recompile with smaller LR

optimizer_fine = Adam(learning_rate=1e-5)
model.compile(optimizer=optimizer_fine,
loss='categorical_crossentropy',
metrics=['accuracy'])

history_finetune = model.fit(
train_generator,
validation_data=val_generator,
epochs=epochs_finetune,
class_weight=class_weights
)

# -------------------------

# EVALUATION

# -------------------------

val_loss, val_acc = model.evaluate(val_generator)
print(f"\nFinal Validation Accuracy: {val_acc * 100:.2f}%")

# Confusion matrix

Y_pred = model.predict(val_generator)
y_pred = np.argmax(Y_pred, axis=1)
print("\nConfusion Matrix:")
print(confusion_matrix(val_generator.classes, y_pred))
print("\nClassification Report:")
print(classification_report(val_generator.classes, y_pred,
target_names=list(train_generator.class_indices.keys())))

# -------------------------

# SAVE MODEL

# -------------------------

model.save("EfficientNet_balanced_model.h5")
print("\nModel saved as 'EfficientNet_balanced_model.h5'")

# -------------------------

# PLOT METRICS

# -------------------------

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'] + history_finetune.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'] + history_finetune.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy Progress')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'] + history_finetune.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'] + history_finetune.history['val_loss'], label='Val Loss')
plt.title('Loss Progress')
plt.legend()

plt.show()
