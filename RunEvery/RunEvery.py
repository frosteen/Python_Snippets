import time


class RunEvery:
    def __init__(self, interval, start_time=time.time()):
        self.interval = interval
        self.previous_time = start_time

    def check(self):
        current_time = time.time()
        if time.time() - self.previous_time >= self.interval:
            self.previous_time = current_time
            return True
        return False
