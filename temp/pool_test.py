from multiprocessing import Pool
import numpy as np


def p():
    img_labels = [i for i in range(20)]
    X = []
    y = []
    pool = Pool()
    results = pool.map(prepare_data, chunks(img_labels, 4))
    for res in results:
        X = np.concatenate((X, res[0]))
        y = np.concatenate((y, res[1]))
    return X, y


def prepare_data(list):
    X = list
    y = list
    return X, y


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in xrange(0, len(lst), n):
        yield lst[i:i + n]


def main():
    z = p()
    print(z)


if __name__ == "__main__":
    main()