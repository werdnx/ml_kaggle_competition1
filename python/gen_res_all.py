import csv
import os
from pathlib import Path

thresholds = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
filename_stat = 'result_stat_model3_EfficientNetB3_gen_300_15_15_13.csv'
DIR = '/Users/dmitrenkoandrey/PycharmProjects/ml_kaggle_competition1/result'


def main():
    Path(DIR + '/result' + '/' + filename_stat).mkdir()
    for threshold in thresholds:
        rows = []
        mal = 0
        bel = 0
        with open(DIR + '/' + filename_stat) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if float(row[2]) > threshold:
                    rows.append((row[0], '1'))
                    mal = mal + 1
                else:
                    rows.append((row[0], '0'))
                    bel = bel + 1

        filename = DIR + '/result/' + filename_stat + '/' + str(threshold) + '.csv'
        # writing to csv file
        if os.path.exists(filename):
            os.remove(filename)
        fields = ['image_name', 'target']
        with open(filename, 'w') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)
            # writing the fields
            csvwriter.writerow(fields)
            # writing the data rows
            csvwriter.writerows(rows)
        print('threashold = ' + str(threshold) + ' belign = ' + str(bel) + ' malign = ' + str(mal))


if __name__ == "__main__":
    main()
