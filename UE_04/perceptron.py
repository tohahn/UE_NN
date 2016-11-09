import numpy as n

def add(w, x, epsilon):
    return w + ((n.dot((-1*w),x)+epsilon)/n.power(n.linalg.norm(x),2))*x

def sub(w, x, epsilon):
    print w - ((n.dot((-1*w),x)+epsilon)/n.power(n.linalg.norm(x),2))*x
    raw_input()
    return w - ((n.dot((-1*w),x)+epsilon)/n.power(n.linalg.norm(x),2))*x

def test(w, x, result, epsilon):
    if (result == True):
        if (n.dot(w,x) > 0):
            return (w,False)
        else:
            return (add(w, x, epsilon),True)
    else:
        if (n.dot(w,x) < 0):
            return (w,False)
        else:
            return (sub(w, x, epsilon),True)

def learn(data, results, epsilon):
    w = n.zeros(data.shape[1])
    num = data.shape[0]
    count = 0

    for i in range(0,10000):
        res = test(w, data[n.random.randint(0, num),:], count, epsilon)
        w = res[0]
        if (res[1]):
            count += 1

    return (w,count)
