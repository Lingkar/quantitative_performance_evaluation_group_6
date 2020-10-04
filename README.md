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

