from module.app import QApplication, APP
import sys, os
from PyQt5 import sip

def main():
    if os.path.exists('temp/result.log'):
        os.remove('temp/result.log')
    app = QApplication(sys.argv)
    window = APP()
    window.show()
    app.exec_()
    app.exit()

if __name__ == '__main__':
    main()
