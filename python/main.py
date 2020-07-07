from convolutional import run_main
import time

def main():
    start = time.time()
    print("--------------------------------Run main!------------------------------------")
    run_main()
    print("--------------------------------End main!------------------------------------")
    end = time.time()
    total = end -start
    print("total time mls:")
    print(total)


if __name__ == "__main__":
    main()
