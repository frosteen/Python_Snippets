from datetime import datetime


def epoch_converter(epoch_time):
    return datetime.fromtimestamp(int(epoch_time) / 1000)
