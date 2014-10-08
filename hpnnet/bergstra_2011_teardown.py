import sys
import shutil
import os

optimizer_dir_in_experiment = os.getcwd()
print "Going to remove", os.path.join(optimizer_dir_in_experiment, "compiledir")
shutil.rmtree(os.path.join(optimizer_dir_in_experiment, "compiledir"))
