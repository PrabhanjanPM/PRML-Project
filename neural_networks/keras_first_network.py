from keras.models import Sequential 
from keras.layers import Dense 

import numpy as np

data  = np.genfromtxt("../features/data", delimiter=',')
label  = np.append(np.zeros(1508), np.ones(1508))
n     = data.shape[0]
data  = np.transpose(np.vstack((np.ones(n),data)))
model = Sequential()
model.add(Dense(3, input_dim=2, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(data, label, epochs=150, batch_size=10)
scores = model.evaluate(data, label)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
test_data = np.genfromtxt("../features/test_data", delimiter=',')
n         = test_data.shape[0]
test_data = np.transpose(np.vstack((np.ones(n),test_data)))
label     = np.append(np.zeros(760), np.ones(950))
model.fit(test_data, label, epochs=150, batch_size=10)
scores = model.evaluate(test_data, label)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


