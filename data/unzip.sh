for rep in 1 2 3
do
for le in 0 20 40
do
for in in 0 20 40
do
  gzip -d ./experiments/repetition_${rep}/le_${le}_in_${in}/*.gz
done
done
done
gzip -d ./*.gz