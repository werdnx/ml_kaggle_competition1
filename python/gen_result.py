import csv

from model_params import model_name


def main():
    filename_stat = '/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1/output/result_stat_' + model_name + '.csv'
    rows = []
    with open(filename_stat) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if float(row[2]) > 0.9:
                rows.append((row[0], '1'))
            else:
                rows.append((row[0], '0'))

    filename = '/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1/output/result_' + model_name + '.csv'
    # writing to csv file
    fields = ['image_name', 'target']
    with open(filename, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        # writing the fields
        csvwriter.writerow(fields)
        # writing the data rows
        csvwriter.writerows(rows)


if __name__ == "__main__":
    main()
