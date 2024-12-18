
import os
from src.preprocess import get_train_data_gen, get_val_data_gen, load_data
from src.model import build_model

train_dir = os.path.join('datasets', 'train')
validation_dir = os.path.join('datasets', 'validation')
model_save_path = os.path.join('models', 'jaw_disease_detection_model.h5')


IMG_HEIGHT = 150
IMG_WIDTH = 150
EPOCHS = 50
BATCH_SIZE = 32

# Data Preparation
train_data = load_data(get_train_data_gen(), train_dir)
val_data = load_data(get_val_data_gen(), validation_dir)

# Model Building
model = build_model(IMG_HEIGHT, IMG_WIDTH)
model.summary()

# Training
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    steps_per_epoch=train_data.samples // BATCH_SIZE,
    validation_steps=val_data.samples // BATCH_SIZE
)

# Save Model
# os.makedirs('models', exist_ok=True)
# model.save(model_save_path)

# print(f"Model saved to {model_save_path}")
