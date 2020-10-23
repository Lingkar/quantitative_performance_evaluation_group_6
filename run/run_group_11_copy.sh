#/!/bin/bash
count=0
index=20
for rep in 1 2  
do
for le in 0 20 40
do
for in in 0 20 40
do
for ks1 in 3 8 
do 
for ks2 in 3 8 
do
if ((count>index));then
./spark-submit --master spark://10.128.0.3:7077 --driver-cores 2  --driver-memory 1G --total-executor-cores 4 --executor-cores 2 --executor-memory 1G --py-files /home/am72ghiassi/bd/spark/lib/bigdl-0.11.0-python-api.zip,/home/am72ghiassi/bd/quantitative_performance_evaluation_group_6/lenet5.py --properties-file /home/am72ghiassi/bd/spark/conf/spark-bigdl.conf --jars /home/am72ghiassi/bd/spark/lib/bigdl-SPARK_2.3-0.11.0-jar-with-dependencies.jar --conf spark.driver.extraClassPath=/home/am72ghiassi/bd/spark/lib/bigdl-SPARK_2.3-0.11.0-jar-with-dependencies.jar --conf spark.executer.extraClassPath=bigdl-SPARK_2.3-0.11.0-jar-with-dependencies.jar  /home/am72ghiassi/bd/quantitative_performance_evaluation_group_6/lenet5.py  --action train --dataPath /home/am72ghiassi/bd/quantitative_performance_evaluation_group_6/data/experiments/repetition_${rep}/le_${le}_in_${in} --kernelSize1 ${ks1} --kernelSize2 ${ks2} --endTriggerNum 20 
fi
echo $count
((count++))
done
done
done
done
done
