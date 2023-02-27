sequence = range(0, 10)


def global_seq_test():
    for element in sequence:
        print(element)
        if element > 5:
            print("end")
            return


def global_seq_test2():
    for element in sequence:
        print(element)


global_seq_test()
global_seq_test2()