from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model

def create_model(x, y, kind):
    model = Sequential()
    model.add(Dense(128, input_shape=(len(x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(set(y)), activation='softmax'))
    
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(x, y, epochs=200, batch_size=5, verbose = 1)
    model.save(f'{kind}.h5')
    print('Model succesfully created.')
    return model

def get_model(x, y, kind):
    try:
        model = load_model(f'{kind}.h5')
    except:
        model = create_model(x, y, kind)
    return model

