import argparse
import csv
import os

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--network-name', type=str, required=True)
parser.add_argument('-d', '--dataset-name', type=str, required=True)
parser.add_argument('-m', '--multilabel', type=bool, default=False)
args = parser.parse_args()

results_dir = os.path.join('work', args.network_name, args.dataset_name)
train_file = os.path.join(results_dir, 'train.csv')
test_file = os.path.join(results_dir, 'test.csv')

with open(train_file, 'r') as csvfile:
        i = rows = loss = prec = acc = rec = err = 0
        reader = csv.reader(csvfile)
        
        print('TRAIN\n')
        if args.multilabel:
            print('Epoch\tLoss\t\tAccuracy\tPrecision\tRecall')
            for row in reader:
                if int(float(row[0])) == i:
                    loss += float(row[1])
                    acc += float(row[2])
                    prec += float(row[3])
                    rec += float(row[4])
                    rows += 1
                else:
                    print('%d\t%.6f\t%.6f\t%.6f\t%.6f' % (i + 1, loss / rows, acc / rows, prec / rows, rec / rows))
                    i += 1
                    loss = float(row[1])
                    acc = float(row[2])
                    prec = float(row[3])
                    rec = float(row[4])
                    rows = 1
            print('%d\t%.6f\t%.6f\t%.6f\t%.6f' % (i + 1, loss / rows, acc / rows, prec / rows, rec / rows))
        else:
            print('Epoch\tLoss\t\tError')
            for row in reader:
                if int(float(row[0])) == i:
                    loss += float(row[1])
                    err += float(row[2])
                    rows += 1
                else:
                    print('%d\t%.6f\t%.6f' % (i + 1, loss / rows, err / rows))
                    i += 1
                    loss = float(row[1])
                    err = float(row[2])
                    rows = 1
            print('%d\t%.6f\t%.6f' % (i + 1, loss / rows, err / rows))

with open(test_file, 'r') as csvfile:
    reader = csv.reader(csvfile)

    print('\nTEST\n')
    if args.multilabel:
        print('Epoch\tLoss\t\tAccuracy\tPrecision\tRecall')
        for row in reader:
            epoch = int(row[0])
            loss = float(row[1])
            acc = float(row[2])
            prec = float(row[3])
            rec = float(row[4])
            print('%d\t%.6f\t%.6f\t%.6f\t%.6f' % (epoch, loss, acc, prec, rec))
    else:
        print('Epoch\tLoss\t\tError')
        for row in reader:
            epoch = int(row[0])
            loss = float(row[1])
            err = float(row[2])
            print('%d\t%.6f\t%.6f' % (epoch, loss, err))
