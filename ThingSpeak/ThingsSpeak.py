import urllib
import json


def read_data(url):
    request = urllib.request.urlopen(url)
    response = request.read()
    data = json.loads(response)

    return data


def write_data(url):
    urllib.request.urlopen(url)
