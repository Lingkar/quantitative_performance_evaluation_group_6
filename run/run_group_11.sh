#/!/bin/bash
for rep in 1 #2 3 4 5
do
for le in 0 20 40
do
for in in 0 20 40
do
for ks1 in 3 8 
do 
for ks2 in 3 8 
do
for epoch in 5 10 20
do
./spark-submit --master spark://10.128.0.3:7077 --driver-cores 2 --driver-memory 3G --executor-cores 2 --executor-memory 3G --py-files /home/am72ghiassi/bd/spark/lib/bigdl-0.11.0-python-api.zip,/home/am72ghiassi/bd/quantitative_performance_evaluation_group_6/lenet5.py --properties-file /home/am72ghiassi/bd/spark/conf/spark-bigdl.conf --jars /home/am72ghiassi/bd/spark/lib/bigdl-SPARK_2.3-0.11.0-jar-with-dependencies.jar --conf spark.driver.extraClassPath=/home/am72ghiassi/bd/spark/lib/bigdl-SPARK_2.3-0.11.0-jar-with-dependencies.jar --conf spark.executer.extraClassPath=bigdl-SPARK_2.3-0.11.0-jar-with-dependencies.jar  /home/am72ghiassi/bd/quantitative_performance_evaluation_group_6/lenet5.py  --action train --dataPath /home/am72ghiassi/bd/quantitative_performance_evaluation_group_6/data/experiments/repetition_${rep}/le_${le}_in_${in} --kernelSize1 ${ks1} --kernelSize2 ${ks2} --endTriggerNum ${epoch} > /home/am72ghiassi/bd/quantitative_performance_evaluation_group_6/outputs/repetition_${rep}/le_${le}_in_${in}_ks1_${ks1}_ks2_${ks2}_epochs_${epoch}.txt
done
done
done
done
done
done
