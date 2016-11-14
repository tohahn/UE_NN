import numpy as n
import perceptron as p
import matplotlib.pyplot as plt
import pandas as pd
from pandas.tools.plotting import parallel_coordinates

if __name__ == "__main__":
    samples = 100
    dimensions = 10
    w = n.random.rand(dimensions) * 20 - 10
    data = n.zeros(samples, dtype=[('data', 'f8', (dimensions)), ('res', 'f8'), ('names', 'a4')])
    data['data'] = n.random.rand(samples,dimensions) * 20 - 10
    print("The original weight vector: {0}".format(w))

    for i in range(0,samples):
        if (n.dot(w, data['data'][i]) > 0):
            data['res'][i] = 5
        elif (n.dot(w, data['data'][i]) < 0):
            data['res'][i] = -5
        else:
            del(data[i])

    res = p.learn(data)
    w = res[0]
    error = 0
    
    print("The learned vector: {0}".format(w))

    for i in range(0,samples):
        if (data['res'][i] > 0 and n.dot(data['data'][i], w) < 0 or data['res'][i] < 0 and n.dot(data['data'][i], w) > 0):
            error += 1
    print("The number of wrong classifications: {0}\nThe number of adjustments made to the vector: {1}\nThe total number of test iterations: {2}".format(error, res[1], res[2]))
    
    for i in range(0,samples):
        if (n.dot(data['data'][i], w) < 0):
            data['names'][i] = "N/-1"
        else:
            data['names'][i] = "P/1"
    
    data_columns = ['var1', 'var2', 'var3', 'var4', 'var5', 'var6', 'var7', 'var8', 'var9', 'var10']
    df = pd.concat([
        pd.DataFrame(data['data'], columns=data_columns),
        pd.DataFrame(data['res'], columns=['orig_result']),
        pd.DataFrame(data['names'], columns=['names'])], axis=1)

    plt.figure()

    parallel_coordinates(df, "names")

    plt.show()
