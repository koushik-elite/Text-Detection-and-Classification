
import os
import numpy as np
import json
import pickle
import csv
import re
from unidecode import unidecode

from sklearn.preprocessing import MultiLabelBinarizer

count=0
csv.field_size_limit(2147483647)
train_reader = csv.reader(open('TRAIN.csv', encoding="Latin-1"), skipinitialspace=True)
test_reader = csv.reader(open('TEST.csv', encoding="Latin-1"), skipinitialspace=True)

next(train_reader)
next(test_reader)

for line in train_reader:
    count = count + 1
    output   = re.sub(' +', ' ', unidecode(str(line[0]).replace('\n', ' ').replace('"', '')))
    np.savetxt('C:\\author_detection\\stories\\{0}_{1}.txt'.format(count, str(line[1])), [output], fmt='%s')

count=0
for line in test_reader:
   count = count + 1
   output   = re.sub(' +', ' ', unidecode(str(line[0]).replace('\n', ' ').replace('"', '')))
   np.savetxt('C:\\author_detection\\test_stories\\{0}_1.txt'.format(count), [output], fmt='%s')