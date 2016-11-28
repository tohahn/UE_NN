import numpy as np
import meta as m
from matplotlib import pyplot as plt

def extractDigits(filename, expected_num):
	data_count = 0
	digit_count = 0
	data_points_per_digit = 192
	data_points_per_line = 12
	
	digits = np.zeros(expected_num, dtype=[('data', 'f', data_points_per_digit), ('value', 'i')])
	
	with open(filename) as f:
		lines = f.readlines()
	
	for i,line in enumerate(lines):
		digits_line = line.split()
		if (len(digits_line) == data_points_per_line):
			for num in digits_line:
				digits['data'][digit_count][data_count] = float(num)
				data_count += 1
		elif (len(digits_line) == 10):
			for i,num in enumerate(digits_line):
				if (num == "1.0"):
					digits['value'][digit_count] = i
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
	name = "./data/digits.trn"
	number = 1000
	digits = extractDigits(name, number)
	
	erwartungswerte = m.erwartung(digits)
	komponenten = m.hauptkomponenten(digits, 9)
	
	plt.gray()
	for i,e in enumerate(erwartungswerte):
		plt.subplot(10,len(erwartungswerte),i+1)
		plt.tick_params(axis='both',  which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
		plt.imshow(np.reshape(e, (16,12)))
		for j,k in enumerate(komponenten[i]):
			plt.subplot(10,len(erwartungswerte),(j+1)*10+i+1)
			plt.tick_params(axis='both',  which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
			plt.imshow(np.reshape(k, (16,12)))
	plt.show()
	
	for d in digits:
		d['value'] = 0
	
	erwartungswert = m.erwartung(digits)[0]
	komponenten = m.hauptkomponenten(digits, 15)[0]
	
	plt.gray()
	plt.subplot(1,16,1)
	plt.tick_params(axis='both',  which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
	plt.imshow(np.reshape(erwartungswert, (16,12)))
	for i,k in enumerate(komponenten):
		plt.subplot(1,16,2+i)
		plt.tick_params(axis='both',  which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
		plt.imshow(np.reshape(k, (16,12)))
	plt.show()
