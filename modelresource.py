import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score
import operator
from functools import reduce
def resouce(filename):
    data = []
    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            eachline = line.split(',')
            read_data = list(map(int, eachline))
            data.append(read_data)
            line = f.readline()
        return data
if __name__ == '__main__':
    test_content = resouce('dogcattest.txt')
    content1 = np.array(test_content)
    test = reduce(operator.add, content1)
    pred_content = resouce('dogcatpredict.txt')
    content2 = np.array(pred_content)
    pred=reduce(operator.add, content2)
    result = confusion_matrix(test, pred)
    print('混淆矩阵：')
    print(result)
    r = recall_score(test, pred)
    print('召回率：')
    print(r)
    acc = accuracy_score(test, pred)
    print('准确率：')
    print(acc)