import csv
import os

from pathlib import Path

thresholds = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.045, 0.046, 0.047, 0.048, 0.049, 0.05, 0.051, 0.052, 0.053, 0.054,
              0.055, 0.056, 0.057, 0.058, 0.059, 0.06, 0.061, 0.062, 0.063, 0.064, 0.065, 0.066, 0.067, 0.068, 0.069,
              0.07, 0.08, 0.09, 0.1, 0.15, 0.2,
              0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]
filename_stat = 'ensembled_b6_40_meta.csv'
DIR = '/Users/dmitrenkoandrey/PycharmProjects/ml_kaggle_competition1/result'


def main():
    Path(DIR + '/result' + '/' + filename_stat).mkdir()
    for threshold in thresholds:
        rows = []
        mal = 0
        bel = 0
        line_count = 0
        with open(DIR + '/' + filename_stat) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if line_count == 0:
                    # print('Column names are ', ", ".join(row))
                    line_count += 1
                else:
                    if float(row[1]) > threshold:
                        rows.append((row[0], '1'))
                        mal = mal + 1
                    else:
                        rows.append((row[0], '0'))
                        bel = bel + 1

        filename = DIR + '/result/' + filename_stat + '/' + str(threshold) + '_' + str(mal) + '.csv'
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
