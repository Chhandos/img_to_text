from pickle import load
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, LSTM, Embedding, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam

# Load data
with open("xception_features.pkl", "rb") as f:
    img_feature_vector = load(f)

with open("img_x_capindex.pkl", "rb") as f:
    img_x_cap = load(f)
    # print(img_x_cap)

with open("word_index.pkl", "rb") as f:
    wordindexdic = load(f)

c=0
for i in wordindexdic :
    if wordindexdic[i]==7243:
        print(i)
        c+=1
print(c)      






# Prepare training data
X1, X2, Y = [], [], []

for path in img_feature_vector:
    image = img_feature_vector[path]
    caption_array = img_x_cap[path]
    for seq in caption_array:
        X1.append(image)
        X2.append(seq[0][:-1])  # input sequence (all but last word)
        Y.append(seq[0][-1])    # target word (last word)

X1 = np.array(X1)  # shape: (num_samples, 2048)
X2 = np.array(X2)  # shape: (num_samples, max_len - 1)
Y = np.array(Y)    # shape: (num_samples,)

max_word_idx = max([max(seq) for seq in X2 if len(seq) > 0])
print("Max word index in X2:", max_word_idx)


# Hyperparameters
vocab_size = 7242     # tokenizer.word_index starts from 1, so vocab indices: 1–7240 (0 for padding)
embedding_dims = 512
max_len = 38         # max caption length-1
learning_rate = 1e-3

# Image branch
img_input = Input(shape=(2048,), name="input_image")
img_dense = Dense(512, activation="relu")(img_input)
img_dropout = Dropout(0.2)(img_dense)

# Text branch
caption_input = Input(shape=(max_len - 1,), name="caption_input")
embedding = Embedding(input_dim=vocab_size + 1, output_dim=embedding_dims, mask_zero=True)(caption_input)
lstm_out = LSTM(512)(embedding)
text_dropout = Dropout(0.2)(lstm_out)

# Merge image and text features
merged = Concatenate()([img_dropout, text_dropout])  # shape: (None, 512)
dense1 = Dense(512, activation="relu")(merged)
dense1_dropout = Dropout(0.2)(dense1)
output = Dense(vocab_size + 1, activation="softmax")(dense1_dropout)

# Define and compile model
model = Model(inputs=[img_input, caption_input], outputs=output)
model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(learning_rate), metrics=["accuracy"])

# Callbacks
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
]

# Train
# model.fit([X1, X2], Y, batch_size=128, epochs=10, validation_split=0.1, callbacks=callbacks)

# Save
# model.save("CaptionModel.keras")
