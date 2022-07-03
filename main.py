import matplotlib.pyplot as plt
import numpy as np


filename = 'loss.txt'

def txt_strtonum_feed(filename):
    data = []
    with open(filename, 'r') as f:  # with语句自动调用close()方法
        line = f.readline()
        while line:
            eachline = line.split(',')  ###按行读取文本文件，每行数据以列表形式返回
            read_data = [float(x) for x in eachline[0:9]]  # TopN概率字符转换为float型
            # lable = [int(x) for x in eachline[-1]]  # lable转换为int型
            # read_data.append(lable[0])
            read_data = list(map(float, eachline))
            data.append(read_data)
            line = f.readline()
        return data


if __name__ == '__main__':
    test_content = txt_strtonum_feed('loss.txt')
    content = np.array(test_content)
    # print(content.size)
    x = np.array([range(0,content.size)])#content的长度
    # print(x)
    plt.title("Loss Function")
    plt.ylabel("Loss Rate")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.plot(x, content,  color="blue")
    plt.scatter(x, content,  color="blue")
    # plt.plot(x, content,color="blue", marker="o")
    plt.legend()
    plt.show()


