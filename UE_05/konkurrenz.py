import numpy as np

def clustering(digits, k):
	dim_num = len(digits['data'][0])
	digits_number = len(digits)
	weight_vecs = np.random.rand(k, dim_num)
	np.apply_along_axis(np.linalg.norm, 1, weight_vecs)
	
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
	
	return weight_vecs

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
	
	for i,v in enumerate(count):
		print("Dem Vektor {0} wurden folgende Ziffern in folgender Haeufigkeit zugeteilt:".format(i))
		for j,freq in enumerate(v):
			print("Ziffer {0}: {1} mal".format(j, freq))
