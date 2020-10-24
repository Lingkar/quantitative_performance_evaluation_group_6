# quantitative_performance_evaluation_group_6

For the data paths to match, clone repository in working directory

```
cd /home/am72ghiassi/bd 
```

```
git clone https://github.com/Lingkar/quantitative_performance_evaluation_group_6.git
```

Additionally, ensure that each of the folders with different data also has the 't10k-labels-idx1-ubyte.gz' and 't10k-images-idx3-ubyte.gz' because test accuracy per epoch is (probably) tested on those sets.

For generating jobs with sparkgen:

place exploratory.json in '/home/am72ghiassi/bd/sparkgen-bigdl/src/sparkgen'

Then, simply execute 

```
./sparkgen -r -d -c ./exploratory.json 
```

and observe the results in the out files in /test as shown in the tutorial


**************************************************************************
This is a modified version of previous lenet5 model.
For the use of lenet5.py, put it under '/home/am72ghiassi/bd/codes/',
and create a directory '/home/am72ghiassi/bd/result_log/' to store the train logs and validation logs.
```
mkdir /home/am72ghiassi/bd/result_log/

```
For the interpretation of results,
train_loss, val_top1, train_throughput of all the training steps will be printed in the form of numpy.array respectively.

There is also a piece of commented out code that plots these three variables against training steps. 
We can discuss whether we need to plot the fig for each experiment.

For the hyper parameters to tune, options.kernelSize1 and options.kernelSize2 are added.
Remember to add the following into the exploratory.json.

```
"--kernelSize1":" ",
"--kernelSize2":" "
```
For the other fixed fixed parameters, the current value:
batchSize:128;
learningRate:0.01;
learningRateDecay:0.0002.

**************************************************************************
### Extract results from output file (example)
cat ./le_0_in_0_ks1_3_ks2_3.txt | grep Top1 | grep -e 'Epoch 5\|Epoch 10\|Epoch 20' | sed -r 's/.*Epoch\s([[:digit:]]*).*accuracy: ([[:digit:]]\.[[:digit:]]*)\)/ \1\t\2 /' 


**************************************************************************
For the use of /labeler/main.py, use the sample option(-ks1:kernelSize1, -ks2:kernelSize2, -e: epochs, -er: label_error_ratio, -in:image_noise)
```
python3 main.py -s mnist -a 1 -b 1 -t 0.1 -ks1 8 -ks2 8 -e 5 -er 40 -in 40
```
The output is a pickle file such as "WSstat_alpha1+beta1+threshold0.1+ks18+ks28+epochs5+error_ratio40+image_noise40.pickle".
To find the results in the pickle file:

```
f = open('WSstat_alpha1+beta1+threshold0.1+ks18+ks28+epochs5+error_ration40+image_noise40.pickle', 'rb')
info = pickle.load(f, encoding="utf8")
val_acc = info[14][0]
train_acc = info[15][0]
val_loss = info[16][0]
train_loss = info[17][0]
time = info[18]
```

Remember to change the datapath used to store the data in line110 to line113 of the main.py
```
X_train = idx2numpy.convert_from_file("/le_0_in_%d/train-images-idx3-ubyte"%image_noise)
X_test = idx2numpy.convert_from_file("/le_0_in_%d/t10k-images-idx3-ubyte"%image_noise)
y_train = idx2numpy.convert_from_file("/le_0_in_%d/train-labels-idx1-ubyte"%image_noise)
y_test = idx2numpy.convert_from_file("/le_0_in_%d/t10k-labels-idx1-ubyte"%image_noise)
```
