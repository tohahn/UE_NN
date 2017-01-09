import numpy as np
from NeuralNet import NeuralNet

def extractDigits(filename, expected_num):
        data_count = 0
        digit_count = 0
        data_points_per_digit = 192
        data_points_per_line = 12
        
        digits = np.zeros(expected_num, dtype=[('data', 'f', data_points_per_digit), ('value', 'f', 10)])
        
        with open(filename) as f:
                lines = f.readlines()
        
        for i,line in enumerate(lines):
                digits_line = line.split()
                if (len(digits_line) == data_points_per_line):
                        for num in digits_line:
                                digits['data'][digit_count][data_count] = float(num)
                                data_count += 1
                elif (len(digits_line) == 10):
                        digits['value'][digit_count] = np.zeros(10)
                        for i,num in enumerate(digits_line):
                                if (num == "1.0"):
                                        digits['value'][digit_count][i] = float(num)
                                        break
                else:
                        if (data_count == data_points_per_digit and digit_count < expected_num):
                                digit_count += 1
                                data_count = 0
                        else:
                                print("Exited because of wrong data")
                                raise SystemExit
        
        if (digit_count == expected_num):
                return digits
        else:
                print("Exited because of few digits")
                raise SystemExit

if __name__ == "__main__":      
        dream_digit = [0.0, 0.5, 0.2, 0.8, 0.2, 1.0, 0.3, 0.7, 0.2, 0.8, 0.6, 1.0, 0.7, 1.0, 0.4, 0.3, 0.2, 0.3, 0.2, 0.8, 0.9, 0.8, 0.6, 0.4, 0.2, 0.7, 0.2, 0.1, 0.8, 0.4, 0.1, 0.7, 0.6, 0.0, 0.9, 1.0, 0.9, 0.2, 0.8, 0.0, 0.2, 0.3, 0.4, 0.3, 0.3, 0.9, 0.5, 0.5, 0.9, 0.0, 0.2, 0.2, 0.8, 0.2, 0.8, 0.9, 0.4, 0.0, 0.9, 0.3, 0.0, 0.4, 0.8, 0.2, 0.9, 0.4, 0.4, 1.0, 0.9, 0.7, 0.8, 0.6, 0.7, 0.9, 0.3, 0.6, 0.5, 0.4, 0.0, 0.8, 0.1, 0.1, 0.3, 0.1, 0.7, 1.0, 0.7, 0.8, 0.5, 0.4, 0.6, 0.8, 0.4, 0.2, 0.8, 0.2, 0.2, 0.4, 0.6, 0.8, 0.1, 0.7, 0.9, 0.7, 0.8, 0.1, 0.1, 0.7, 0.9, 0.5, 0.2, 0.0, 0.6, 0.9, 0.9, 1.0, 0.3, 0.5, 0.9, 0.4, 0.4, 0.7, 0.0, 0.7, 0.7, 0.0, 0.6, 0.1, 0.7, 0.4, 0.2, 0.5, 0.8, 0.9, 0.6, 0.4, 0.0, 0.2, 0.6, 0.0, 0.9, 0.9, 0.5, 0.0, 1.0, 0.4, 0.6, 0.1, 0.5, 1.0, 0.2, 0.1, 0.6, 0.6, 0.6, 0.4, 0.1, 0.3, 1.0, 0.2, 0.1, 0.2, 0.3, 0.5, 0.4, 0.6, 0.1, 0.7, 0.2, 0.3, 0.3, 0.1, 0.1, 0.2, 0.3, 0.4, 0.2, 1.0, 0.7, 0.7, 0.4, 1.0, 0.2, 1.0, 0.8, 0.2, 1.0, 0.7, 0.3, 0.2, 0.0, 0.9]
        
        train_name = "../data/digits.trn"
        train_number = 1000
        train_digits = extractDigits(train_name, train_number)
        
        test_name = "../data/digits.tst"
        test_number = 200
        test_digits = extractDigits(test_name, test_number)
        
        myNet = NeuralNet(192,10,[20,20,20])
        myNet.train(train_digits['data'], train_digits['value'], test_digits['data'], test_digits['value'])
    
        myNet.dream(dream_digit);
