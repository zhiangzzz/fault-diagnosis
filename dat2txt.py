import os

data_dir = '/home/rapperli/practice_project/SVDD_faultdiagnosis/ceshi1/'

files = os.listdir(data_dir)
n = 0
for i in files:
    oldname = data_dir +files[n]
    newname = data_dir + 'test'+str(n) + '.csv'
    os.rename(oldname,newname)

    n+=1