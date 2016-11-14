import numpy as n

def add(w, x, epsilon):
    return w + ((n.dot(-1*w, x)+epsilon)/n.power((n.linalg.norm(x)), 2))*x

def sub(w, x, epsilon):
    return w - ((n.dot(1*w, x)+epsilon)/n.power((n.linalg.norm(x)), 2))*x

def test(w, x, epsilon):
    if (x['res'] > 0):
        if (n.dot(w,x['data']) > 0):
            return (w,False)
        else:
            return (add(w, x['data'], epsilon),True)
    else:
        if (n.dot(w,x['data']) < 0):
            return (w,False)
        else:
            return (sub(w, x['data'], epsilon),True)

def learn(data):
    w = n.zeros(data['data'].shape[1])
    num = data.shape[0]
    wrong_count = 0
    epsilon = 0.001
    convergence = 0
    total_count = 0

    while (convergence < 1000):
        random_int = n.random.randint(0, num)
        res = test(w, data[random_int], epsilon)
        w = res[0]
        total_count += 1
        convergence += 1
        if (res[1]):
            wrong_count += 1
            convergence = 0

    return (w,wrong_count,total_count)
