import numpy as np

def erwartung(digits):
	num_digits = len(set(digits['value']))
	erwartungswerte = np.zeros((num_digits,len(digits['data'][0])))
	frequency = np.zeros(num_digits)
	
	for d in digits:
		erwartungswerte[d['value']] += d['data']
		frequency[d['value']] += 1
	
	for i,e in enumerate(erwartungswerte):
		erwartungswerte[i] = e / frequency[i]
	
	return erwartungswerte
	
def hauptkomponenten(digits, amount):
	komponenten = np.random.rand(len(set(digits['value'])),amount,len(digits['data'][0]))
	lernrate = 1
	amount = len(digits)
	for c in komponenten:
		for k in c:
			k /= np.linalg.norm(k)
	
	while (lernrate > 0):
		example = digits[np.random.randint(0,amount)]
		example_class = example['value']
		example_data = example['data']
		for i,k in enumerate(komponenten[example_class]):
			scalar = np.dot(example_data,k)
			komponenten[example_class][i] = k + lernrate * scalar * (example_data - scalar * k)
			komponenten[example_class][i] /= np.linalg.norm(komponenten[example_class][i])
			example_data -= scalar * komponenten[example_class][i]
		lernrate -= 0.001
	
	return komponenten
