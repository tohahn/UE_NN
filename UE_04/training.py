import numpy as n
import perceptron as p

if __name__ == "__main__":
    dimensions = 10
    samples = 100
    epsilon = 0.001

    w = n.random.rand(10) * 20 - 10
    data = n.random.rand(samples,dimensions) * 20 - 10
    results = n.zeros(samples)

    for i in range(0,samples):
        if (n.dot(w, data[i]) > 0):
            results[i] = 1
        else:
            results[i] = -1

    res = p.learn(data, results, epsilon)
    w = res[0]
    print(res)

    for i in range(0,samples):
        print("Original result: {0} / Trained Result: {1}".format(results[i], n.dot(w, data[i])))
