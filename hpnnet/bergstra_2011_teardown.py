import sys
import shutil
import os

optimizer_dir_in_experiment = sys.argv[1]
print "Going to remove", os.path.join(optimizer_dir_in_experiment, "compiledir")
shutil.rmtree(os.path.join(optimizer_dir_in_experiment, "compiledir"))
