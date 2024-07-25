import threading
import time
class MyThread(threading.Thread):
    def __init__(self, thread_name):
        # 注意：一定要显式的调用父类的初始化函数。
        super(MyThread, self).__init__(name=thread_name)

    def run(self):
        while 1:
            time.sleep(1)
            print("%s正在运行中......" % self.name)

if __name__ == '__main__':
    for i in range(10):
        MyThread("thread-" + str(i)).start()