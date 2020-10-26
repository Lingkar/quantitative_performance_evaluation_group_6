#/!/bin/bash
echo -e "rep\tle\tnoise\tks1\tks2\tepochs\tacc" 
for rep in 1 2 3 4 5 
do
for le in 0 20 40
do
for in in 0 20 40
do
for ks1 in 3 8 
do 
for ks2 in 3 8 
do
cat ./repetition_${rep}/le_${le}_in_${in}_ks1_${ks1}_ks2_${ks2}.txt | grep Top1 | grep -e 'Epoch 5\|Epoch 10\|Epoch 20' | sed -r "s/.*Epoch\s([[:digit:]]*).*accuracy: ([[:digit:]]\.[[:digit:]]*)\)/ $rep\t$le\t$in\t$ks1\t$ks2\t\1\t\2 /"
done
done
done
done
done
