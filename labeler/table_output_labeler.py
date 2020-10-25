import pickle
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import os
import re

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

root_dir = r"" # put all the pickle files to this directory
val_acc = []
avg_val_acc=[]
val_loss=[]
label_error=[]
image_noise=[]
labeler=[]
repetition=[]
for file in os.listdir(root_dir):
    file_name = root_dir + file
    filein = open(file_name, "rb")
    info = pickle.load(filein, encoding="utf8")
    val_acc.extend(info[14][0])
    s_bnn = 0
    for i in range(0, 10):
        s_bnn = s_bnn + val_acc[-5 * i - 1]
    average_valacc = s_bnn / 10
    avg_val_acc.append(average_valacc)
    val_loss.extend(info[16][0])
    label_error.append(re.findall('\d+', file)[9])
    image_noise.append(re.findall('\d+', file)[10])
    labeler.append(re.search(r'True|False', file).group())
    repetition.append(re.findall('\d+', file)[11])

    filein.close()

anova_table = pd.DataFrame({'val_acc':avg_val_acc,
                            'label_error':label_error,
                            'image_noise':image_noise,
                            'labeler':labeler,
                            'repetition':repetition})
print(anova_table)





