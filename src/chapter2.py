import numpy as np

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y


if __name__ == "__main__":
    print("\ntest AND")
    for x1 in [0, 1]:
        for x2 in [0, 1]:
            print(f"{x1} AND {x2}: {AND(x1, x2)}")
    
    print("\ntest NAND")
    for x1 in [0, 1]:
        for x2 in [0, 1]:
            print(f"NOT {x1} AND {x2}: {NAND(x1, x2)}")
    
    print("\ntest OR")
    for x1 in [0, 1]:
        for x2 in [0, 1]:
            print(f"{x1} OR {x2}: {OR(x1, x2)}")
    
    print("\ntest XOR")
    for x1 in [0, 1]:
        for x2 in [0, 1]:
            print(f"{x1} XOR {x2}: {XOR(x1, x2)}")
