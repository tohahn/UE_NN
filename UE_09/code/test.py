import numpy as np
from keras.models import load_model
from keras.models import model_from_json

if __name__ == "__main__":
    test_in_file = open("../data/in.tst", "r")
    test_in_string = np.fromstring(test_in_file.read(), "int8") - 48
    test_in_file.close()

    test_out_file = open("../data/out.tst", "r")
    test_out_string = np.fromstring(test_out_file.read(), "int8") - 48
    test_out_file.close()

    dim = len(test_in_string) - 11
    test_in = np.zeros((dim, 11, 1))
    test_out = np.zeros((dim, 1))
    
    for i in range(0, len(test_in_string)-11):
        test_in[i] = np.reshape(test_in_string[i:i+11], (11,1))
        test_out[i] = test_out_string[i+11]
    
    model = model_from_json(open("model.json", "r").read())
    model.load_weights("model.h5")
    predictions = model.predict(test_in)
    
    predictions_file = open("predictions.out", "w");
    predictions_file.write("".join([str(predictions[x][0]) + "," for x in range(0, len(predictions))]))
    predictions_file.close()
