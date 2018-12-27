import threading
from PyQt5.QtCore import QThread, pyqtSignal

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


class MyQThread(QThread):
    _signal = pyqtSignal(str)
    def __init__(self, func):
        super(MyQThread, self).__init__()
        self.func = func
        self._pause_flag = threading.Event()
        self._run_flag = threading.Event()

    def run(self):
        r = self.func()
        self._signal.emit(r)


