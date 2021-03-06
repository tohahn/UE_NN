import numpy as np
from texttable import Texttable

def clustering(digits, k):
	dim_num = len(digits['data'][0])
	digits_number = len(digits)
	weight_vecs = np.random.rand(k, dim_num)
	for i,v in enumerate(weight_vecs):
		weight_vecs[i] = v / np.linalg.norm(v)
	
	for i in range(0,100000):
		current_instance = digits['data'][np.random.randint(0,digits_number)]
		largest_index = -1
		largest_value = float("-inf")
		
		for j in range(0,k):
			current_value = np.dot(current_instance, weight_vecs[j])
			if (current_value > largest_value):
				largest_value = current_value
				largest_index = j
		
		temp_vector = weight_vecs[largest_index] + current_instance
		weight_vecs[largest_index] = temp_vector / np.linalg.norm(temp_vector)
	
	return assign_numbers(digits, weight_vecs)

def assign_numbers(digits, weight_vecs):
	count = np.zeros((len(weight_vecs), 10))
	
	for digit in digits:
		largest_index = -1
		largest_value = float("-inf")
		
		for j in range(0,len(weight_vecs)):
			current_value = np.dot(digit['data'], weight_vecs[j])
			if (current_value > largest_value):
				largest_value = current_value
				largest_index = j
				
		count[largest_index][digit['value']] += 1
	
	named_vecs = np.zeros(len(weight_vecs), dtype=[('vector', 'f', len(weight_vecs[0])), ('digit', 'i')])
	
	for i,v in enumerate(named_vecs):
		named_vecs['vector'][i] = weight_vecs[i]
		named_vecs['digit'][i] = np.argmax(count[i])
		
	print(named_vecs['digit'])
	
	return named_vecs

def predict_number(digit, named_vecs):
	largest_index = -1
	largest_value = float("-inf")

	for j in range(0,len(named_vecs)):
		current_value = np.dot(digit, named_vecs['vector'][j])
		if (current_value > largest_value):
			largest_value = current_value
			largest_index = j
	
	return named_vecs['digit'][largest_index]

def calc_error(digits, named_vecs):
	error = 0
	
	for d in digits:
		if (d['value'] != predict_number(d['data'], named_vecs)):
			error += 1
	
	return error


def print_results(digits, weight_vecs):
	count = np.zeros((len(weight_vecs), 10))
	
	for digit in digits:
		largest_index = -1
		largest_value = float("-inf")
		
		for j in range(0,len(weight_vecs)):
			current_value = np.dot(digit['data'], weight_vecs[j])
			if (current_value > largest_value):
				largest_value = current_value
				largest_index = j
				
		count[largest_index][digit['value']] += 1
	
	t = Texttable()
	t.add_row(["Vektor", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
	for i,v in enumerate(count):
		t.add_row(np.insert(v, 0, i))
	print t.draw()
