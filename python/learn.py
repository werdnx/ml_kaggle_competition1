import time
from gen_learn2 import run


def main():
    start = time.time()
    print("--------------------------------Run main!------------------------------------")
    run()
    print("--------------------------------End main!------------------------------------")
    end = time.time()
    total = end - start
    print("total time mls:")
    print(total)


if __name__ == "__main__":
    main()
