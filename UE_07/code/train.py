import numpy as np
from neural_network import NN

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
	train_name = "../data/digits.trn"
	train_number = 1000
	train_digits = extractDigits(train_name, train_number)
	
	test_name = "../data/digits.tst"
	test_number = 200
	test_digits = extractDigits(test_name, test_number)
	
	for i in range(0,11):
		nn = NN([192, i*10, 10])
		nn.evaluate(train_digits['data'], train_digits['value'], test_digits['data'], test_digits['value'])
