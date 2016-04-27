import numpy as np
import sys
import pickle
import datetime
import matplotlib.pyplot as plt
import os
import csv
from datetime import datetime, date, time, timedelta
from ast import literal_eval


def playLog(fname='logfile', delim=','):
important = []
keep_phrases = ["test",
              "important",
              "keep me"]

with open(infile) as f:
    f = f.readlines()

for line in f:
    for phrase in keep_phrases:
        if phrase in line:
            important.append(line)
            break

print(important)
