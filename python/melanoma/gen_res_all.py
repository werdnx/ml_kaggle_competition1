import csv
import os

from pathlib import Path

thresholds = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.045, 0.048, 0.05, 0.055, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]
filename_stat = 'submission2_4fold_efnet_5_iter1__fold-1-512x512.model.csv'
DIR = '/result'


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
