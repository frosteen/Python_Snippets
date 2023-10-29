import inspect
import time
from threading import Thread, Lock

count = 0
lock = Lock()


def incre():
    global count
    caller = inspect.getouterframes(inspect.currentframe())[1][3]
    print("Inside %s()" % caller)
    print("Acquiring lock")
    with lock:
        print("Lock Acquired")
        count += 1
        time.sleep(2)


def bye():
    while count < 5:
        incre()


def hello_there():
    while count < 5:
        incre()


hello = Thread(target=hello_there)
hello.start()
goodbye = Thread(target=bye)
goodbye.start()
