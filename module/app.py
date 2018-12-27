from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QCoreApplication, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from gui.gui import Ui_MainWindow
import threading, os, time
import numpy as np
import pandas as pd
from module.gene_data_and_train_net import train_classifer, gene_data,writeAndPrintLog
from module.virtual_cell import load_data, CellSchedule
from module.mthread import MyThread, MyQThread


class APP(QMainWindow, Ui_MainWindow, threading.Thread):
    _translate = QCoreApplication.translate
    disp_text_signal = pyqtSignal(str)
    def __init__(self):
        super(APP, self).__init__()
        self.setupUi(self)
        self.setButton()
        self.initUI()
        self.disp_text_signal.connect(self.display_log)

    def initUI(self):
        self.display_log("请输入参数，多个数据的输入请使用英文','分开，如：6,10")
        self.cancel_order.setText(self._translate("MainWindow", "3,7"))
        self.weights.setText(self._translate("MainWindow", "0.9,0.1"))
        self.cell_size.setText(self._translate("MainWindow", "6"))
        self.seed.setText(self._translate("MainWindow", "100"))
        self.CR.setText(self._translate("MainWindow", "0.6"))
        self.Np.setText(self._translate("MainWindow", "100"))
        self.Gm.setText(self._translate("MainWindow", "10"))
        self.rt.setText(self._translate("MainWindow", "50"))
        self.strategy.setText(self._translate("MainWindow", "greedy"))
        self.pnum.setText(self._translate("MainWindow", "20"))
        self.mnum.setText(self._translate("MainWindow", "30"))
        self.mcls.setText(self._translate("MainWindow", "10"))
        t = MyThread(func=self.drawImg)
        t.setDaemon(True)
        t.start()
        if os.path.exists('temp/analy.png'):
            os.remove('temp/analy.png')

    def setButton(self):
        self.btn_cls.clicked[bool].connect(self.btnfunc_train_classifer)
        self.btn_sche.clicked[bool].connect(self.btnfunc_schedule)
        self.btn_gene_data.clicked[bool].connect(self.btnfunc_genedata)

    def btnfunc_train_classifer(self):
        cls = self.cell_size.text()
        seed = self.seed.text()
        if cls == '' or seed == '':
            text = "虚拟单元大小及训练使用的随机种子不能为空"
            self.display_log(text, clear=True)
            return
        cls = int(cls)
        seed = float(seed)
        text = "工件聚类器训练中...\n虚拟单元大小为{}，随机种子为{}".format(cls, seed)
        self.display_log(text)
        t = threading.Thread(target=train_classifer, args=(cls, seed))
        t.setDaemon(True)
        t.start()
        t.join()
        with open('temp/result.log', 'r') as f:
            text = f.read()
        self.display_log(text)

    def btnfunc_genedata(self):
        pnum = self.pnum.text()
        mnum = self.mnum.text()
        mcls = self.mcls.text()
        if pnum == '' and mnum == '' and mcls == '':
            text = '工件数量，机器数量，机器种类不能为空'
            self.display_log(text, clear=True)
            return
        gene_data(int(mcls), int(mnum), int(pnum))

    def btnfunc_schedule(self):
        paras = self.getInputs()
        text = '调度参数设置如下：\n' + str(paras)
        self.display_log(text, clear=True)
        t = threading.Thread(target=self.shedule, args=(paras,))
        t.setDaemon(True)
        t.start()

    def display_log(self, newtext, clear=False):
        if clear:
            text = newtext
        else:
            oldtext = self.display.toPlainText()
            text =  oldtext + newtext + '\n\n'
        self.display.setText(self._translate("MainWindow", text))

    def shedule(self, paras):
        def dispfunc(text):
            self.disp_text_signal.emit(text)
        transform_time = np.array(pd.read_csv('data/transform_time.csv', header=None))
        ms_process_t, m_per_cls_num = load_data('data/machines.csv', flatten=True)
        writeAndPrintLog('机器加工时间:\n{}'.format(ms_process_t), dispfunc)
        parts, part_process_num = load_data('data/parts.csv')
        writeAndPrintLog('工件:\n{}'.format(parts), dispfunc)

        c = CellSchedule(parts=parts,
                         machine_process_t=ms_process_t,
                         m_cls_num=m_per_cls_num,
                         transform_time=transform_time,
                         paras=paras,
                         name='schedule')
        writeAndPrintLog('每个工件的工序数:\n{}'.format(c.process_num), dispfunc)
        writeAndPrintLog('每道工序的可选机器数:\n{}'.format(c.spare_machine_num), dispfunc)
        if os.path.exists('temp/result.log'):
            os.remove('temp/result.log')
        c.schedule(dispfunc)

    def drawImg(self):
        time.sleep(0.5)
        if os.path.exists('temp/analy.png'):
            with open('temp/analy.png', 'rb') as f:
                img = f.read()
        else:
            with open('temp/preanaly.png', 'rb') as f:
                img = f.read()
        image = QImage.fromData(img)
        pixmap = QPixmap.fromImage(image)
        self.img_analysis.setPixmap(pixmap)

    def getInputs(self):
        cell_size = self.cell_size.text()
        CR = self.CR.text()
        Np = self.Np.text()
        Gm = self.Gm.text()
        RT = self.rt.text()
        weights = self.weights.text().split(',')
        cancel_order = self.cancel_order.text().split(',')
        strategy = self.strategy.text()
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

