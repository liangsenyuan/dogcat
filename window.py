
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
import cv2
from PIL import Image


from tool import get_Wpredict, model


class MainWindow(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon('work.jpg')) #任务栏图片&
        self.setWindowTitle('猫狗识别系统')
        self.to_predict_name = "login.jpg" #初始默认图片
        self.resize(900, 500) #窗口大小
        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        font = QFont('楷体', 17) #设置字体：请放入图片，输出结果
        font1 = QFont('楷体', 13)#设置字体：选择文件，开始识别
        font2 = QFont('黑体', 30)#设置字体：输出结果文字
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        img_title = QLabel("请放入图片：")#设置文本框：提示效果
        img_title.setFont(font)#设置字体
        img_title.setAlignment(Qt.AlignCenter)# 水平方向居中对齐
        self.img_label = QLabel()#读取图片
        img_init = cv2.imread(self.to_predict_name)#设置待输入默认图片
        img_init = cv2.resize(img_init, (400, 400))#设置大小
        cv2.imwrite('1.jpg', img_init)#保存图片到1.jpg上
        self.img_label.setPixmap(QPixmap('1.jpg'))
        left_layout.addWidget(img_title)
        left_layout.addWidget(self.img_label, 1, Qt.AlignCenter)
        left_layout.setAlignment(Qt.AlignCenter)
        left_widget.setLayout(left_layout)
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        btn_change = QPushButton(" 选择图片 ")
        btn_change.clicked.connect(self.change_img)
        btn_change.setFont(font1)#设置字体
        btn_predict = QPushButton(" 开始识别 ")
        btn_predict.clicked.connect(self.predict_img)
        btn_predict.setFont(font1)#设置字体
        label_result = QLabel('识 别 结 果 ')
        label_result.setFont(font)
        self.result = QLabel("")
        self.result.setFont(font2)#输出结果字体
        right_layout.addStretch()
        right_layout.addWidget(label_result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(self.result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(btn_change)
        right_layout.addWidget(btn_predict)
        right_layout.addStretch()
        right_widget.setLayout(right_layout)
        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
        main_widget.setLayout(main_layout)
        #功能实现界面
        self.addTab(main_widget, '功能实现')
        self.setTabIcon(0, QIcon('2.jpg'))#主页面小图标

    def change_img(self):
        #第3个参数：''则为E:\1\project，'data/test/'则为E:\1\project\data\test
        #第4个参数：限制图片类型
        openfile_name = QFileDialog.getOpenFileName(self, '选择文件', '', 'Image files(*.jpg , *.png)')
        img_name = openfile_name[0]
        if img_name == '':
            pass
        else:
            #读取图片
            self.to_predict_name = img_name
            img_init = cv2.imread(self.to_predict_name)
            img_init = cv2.resize(img_init, (400, 400))

            #保存图片
            cv2.imwrite('1.jpg', img_init)#图片保存位置
            self.img_label.setPixmap(QPixmap('1.jpg'))

    def predict_img(self):
        folder = r""
        file_path = folder + "1.jpg"
        img_init = cv2.imread(file_path)
        img_init = cv2.resize(img_init, (400, 400))
        cv2.imwrite('1.jpg', img_init)
        self.img_label.setPixmap(QPixmap('1.jpg'))
        pil_im = Image.open(file_path, 'r')
        result = get_Wpredict(pil_im, model)  # 识别图片，并返回猫狗识别结果
        self.result.setText(result) #窗口输出图片识别结果








        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    x = MainWindow()
    x.show()
    sys.exit(app.exec_())



