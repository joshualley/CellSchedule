from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtCore
from PyQt5.QtGui import QImage, QPixmap
from gui.gui import Ui_MainWindow
import threading, sys, os, time
import numpy as np
import pandas as pd
from module.gene_data_and_train_net import train_classifer
from module.virtual_cell import load_data, CellSchedule

class MyThread(threading.Thread):

    def __init__(self, func, *arg):
        super().__init__()
        self.func = func
        self.arg = arg
        self.__flag = threading.Event()  # 用于暂停线程的标识
        self.__flag.set()  # 设置为True
        self.__running = threading.Event()  # 用于停止线程的标识
        self.__running.set()

    def run(self):
        while self.__running.isSet():
            self.__flag.wait()
            if self.arg != ():
                self.func(*self.arg)
            else:
                self.func()

    def pause(self):
        self.__flag.clear()

    def resume(self):
        self.__flag.set()

    def destroy(self):
        self.__flag.set()  # 将线程从暂停状态恢复, 如何已经暂停的话
        self.__running.clear()


class APP(QMainWindow, Ui_MainWindow, threading.Thread):
    _translate = QtCore.QCoreApplication.translate

    def __init__(self):
        super(APP, self).__init__()
        self.setupUi(self)
        self.setButton()
        self.initUI()
        self.thread = None

    def setButton(self):
        self.btn_cls.clicked[bool].connect(self.btnfunc_train_classifer)
        self.btn_sche.clicked[bool].connect(self.btnfunc_schedule)

    def initUI(self):
        self.display.setText(self._translate("MainWindow", "请输入参数，多个数据的输入请使用英文','分开，如：6,10"))
        self.cancel_order.setText(self._translate("MainWindow", "3,7"))
        self.weights.setText(self._translate("MainWindow", "0.9,0.1"))
        self.cell_size.setText(self._translate("MainWindow", "6,10"))
        self.CR.setText(self._translate("MainWindow", "0.6"))
        self.Np.setText(self._translate("MainWindow", "100"))
        self.Gm.setText(self._translate("MainWindow", "10"))
        self.rt.setText(self._translate("MainWindow", "50"))
        self.strategy.setText(self._translate("MainWindow", "greedy"))
        t = MyThread(func=self.drawImg)
        t.setDaemon(True)
        t.start()


    def btnfunc_train_classifer(self):
        paras = self.cell_size.toPlainText().split(',')
        if len(paras) == 1 and paras[0] != '':
            cls = int(paras[0])
            seed = 8
        elif len(paras) == 2:
            cls = int(paras[0])
            seed = int(paras[1])
        else:
            cls = 6
            seed = 8
        text = "工件分类器训练中。。。\n虚拟单元大小为{}，随机种子为{}".format(cls, seed)
        self.display.setText(self._translate("MainWindow", text))
        t = threading.Thread(target=train_classifer, args=(cls, seed))
        t.setDaemon(True)
        t.start()


    def btnfunc_schedule(self):
        paras = self.getInputs()
        text = '调度参数设置如下：\n' + str(paras)
        self.display.setText(self._translate("MainWindow", text))
        t = threading.Thread(target=self.shedule, args=(paras,))
        t.setDaemon(True)
        t.start()



    def shedule(self, paras):
        transform_time = np.array(pd.read_csv('data/transform_time.csv', header=None))
        # ---------------------------------------------------------------------------------#
        ms_process_t, m_per_cls_num = load_data('data/machines.csv', flatten=True)  #
        print('机器加工时间:\n', ms_process_t)  #
        # ---------------------------------------------------------------------------------#
        parts, part_process_num = load_data('data/parts.csv')
        print('工件:\n', parts)

        c = CellSchedule(parts=parts,
                         machine_process_t=ms_process_t,
                         m_cls_num=m_per_cls_num,
                         transform_time=transform_time,
                         paras=paras,
                         name='schedule')
        print('每个工件的工序数:\n', c.process_num)
        print('每道工序的可选机器数:\n', c.spare_machine_num)
        c.schedule()

    def drawImg(self):
        if os.path.exists('temp/analy.png'):
            time.sleep(0.5)
            with open('temp/analy.png', 'rb') as f:
                img = f.read()
            image = QImage.fromData(img)
            pixmap = QPixmap.fromImage(image)
            self.img_analysis.setPixmap(pixmap)

    def getInputs(self):
        cell_size = self.cell_size.toPlainText().split(',')[0]
        CR = self.CR.toPlainText()
        Np = self.Np.toPlainText()
        Gm = self.Gm.toPlainText()
        RT = self.rt.toPlainText()
        weights = self.weights.toPlainText().split(',')
        cancel_order = self.cancel_order.toPlainText().split(',')
        strategy = self.strategy.toPlainText()
        if cell_size == '' and CR == '' and Np == '' and Gm == '' and RT == '' \
                and weights == [''] and cancel_order == [''] and strategy == '':
            return None
        else:
            # 设置ADE参数
            paras = {
                'CR': float(CR),
                'Np': int(Np),
                'Gm': int(Gm),
                'cell_size': int(cell_size),
                'reschedule_time': int(RT),
                'weight': [float(weights[0]), float(weights[1])],
                'strategy': strategy,  # 'greedy'  or  'anneal'
                'cancel_order': [int(order) for order in cancel_order],  # 重调度时取消加工的工件（可在1~15任选）
            }
            return paras

def main():
    app = QApplication(sys.argv)
    window = APP()
    window.show()
    app.exec_()
    app.exit()

if __name__ == '__main__':
    main()