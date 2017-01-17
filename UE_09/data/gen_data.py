import random

class Counter():
    def __init__(self):
        self.state = -1

    def count(self, i):
        if (i == 1):
            self.state = 0
        elif (i == 0 and self.state != -1):
            self.state += 1
        if (self.state == 10):
            self.state = -1
            return 1
        return 0

if __name__ == "__main__":
    buffer_in = []
    buffer_out = []
    input_file = open("./in.dat", "w")
    output_file = open("./out.dat", "w")
    input_file.truncate()
    output_file.truncate()
    count = Counter()

    for i in range(0,1000000):
        rand = int(bool(random.getrandbits(1)))
        buffer_in.append(rand)
        buffer_out.append(count.count(rand))
        if (len(buffer_in) == 10000):
            input_file.write("".join([str(val) for val in buffer_in]))
            output_file.write("".join([str(val) for val in buffer_out]))
            buffer_in = []
            buffer_out = []

    output_file.close()
    input_file.close()
