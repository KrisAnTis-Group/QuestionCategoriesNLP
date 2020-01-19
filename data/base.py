import numpy as np
import collections
import re

TOKEN_RE = re.compile(r'[\w\d]+')
CLEAR_FOR_DELIMITER = re.compile(r'\,.+\,')

def get_DataSet_on_numpy(path = "DataSet/intent_", subset = "train", delimiter=","):
    line_target = []
    line_data = []
    
    first_open = False;

    f = open(path + subset + ".csv")
    for line in f:
        if first_open:
            line = line.lower()
            columns = line.split(delimiter)
            line_target.append(columns[-1])
            w = columns[1:-1]
            s = ""
            for q in w:
                s += q
            line_data.append(s)
        first_open = True

    dataset = {}
    dataset['target'] = line_target
    dataset['data'] = line_data


    #with open(path + subset + ".csv") as f:
     #   lines = (line for line in f if not line.startswith('#'))
     #   dataset = np.loadtxt(lines, 'str', delimiter=delimiter, skiprows=1)
        
    #data = None
    #data.data = dataset[:,1]
    #data.target = dataset[:,2]

    return dataset