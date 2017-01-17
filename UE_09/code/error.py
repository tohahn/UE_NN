if __name__ == "__main__":
    true_out_file = open("../data/out.tst", "r")
    true_out_string = true_out_file.read()
    true_out_file.close()
    true_out = [float(x) for x in list(true_out_string)[11:]]

    pred_out_file = open("predictions.out", "r")
    pred_out_string = pred_out_file.read()[:-1]
    pred_out_file.close()
    pred_out = [float(x) for x in pred_out_string.split(',')]

    num = len(true_out)
    error = 0

    for x,y in zip(pred_out, true_out):
        error += (x - y)**2
    print("Mean squared error: {0}".format(error/num))
