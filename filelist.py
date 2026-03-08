import os
import csv

f= open('./data/frames/annotation.csv','r+')
w=csv.writer(f)
for files in os.walk('./data/frames/jud_suess'):
    for filename in files:
        w.writerow([filename])