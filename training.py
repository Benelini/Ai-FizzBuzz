import json
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

with open('data.json', 'r') as file:
    data_list = json.load(file)

X = np.array([item['input'] for item in data_list])
y = np.array([item['output'] for item in data_list])

X_engineered = np.array([[x % 3, x % 5, x % 15] for x in X])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_engineered)

label_mapping = {'Fizz': 0, 'Buzz': 1, 'FizzBuzz': 2, 'Number': 3}
y_numeric = np.array(
    [label_mapping.get(label, 3) if label in label_mapping else 3 for label in y])
y_one_hot = to_categorical(y_numeric, num_classes=4)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_one_hot, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(16, activation='relu'))
model.add(Dense(4, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20, batch_size=32)

loss, accuracy = model.evaluate(X_test, y_test)
model.save('fizzbuzz_model')
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%")
