import os
import json

config_data = {}

def load(config_file):
    global config_data
    with open(config_file) as config_data:
        config_data = json.load(config_data)


def get(key):
    global config_data
    val = config_data.get(key, None)
    print("Config.get(%s)=%s" % (key,val))
    if val == None:
        print("WARNING: %s is set to None"%key)
    return val

def set(key, value):
    global config_data
    config_data[key] = value

def dump(json_file):
    global config_data
    with open(json_file, 'w') as data_handler:
        json.dump(config_data, data_handler)