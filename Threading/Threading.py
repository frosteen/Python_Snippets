import time
from threading import Thread


def First():
    while True:
        print("This is First function")
        time.sleep(1)


def Second():
    while True:
        print("This is Second function")
        time.sleep(1)


First_Thread = Thread(target=First, daemon=True)
First_Thread.start()

Second()
