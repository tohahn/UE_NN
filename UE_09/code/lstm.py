import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

if __name__ == "__main__":
    train_in_file = open("../data/in.trn", "r")
    train_in_string = np.fromstring(train_in_file.read(), "int8") - 48
    train_in_file.close()

    train_out_file = open("../data/out.trn", "r")
    train_out_string = np.fromstring(train_out_file.read(), "int8") - 48
    train_out_file.close()
    
    dim = len(train_in_string) - 11
    train_in = np.zeros((dim, 11, 1))
    train_out = np.zeros((dim, 1))
    
    for i in range(0, len(train_in_string)-11):
        train_in[i] = np.reshape(train_in_string[i:i+11], (11,1))
        train_out[i] = train_out_string[i+11]

    model = Sequential()
    model.add(LSTM(1, input_shape=(11, 1)))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.fit(train_in, train_out, nb_epoch=1, batch_size=1, verbose=2)

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
            json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights("model.h5")
            print("Saved model to disk")
