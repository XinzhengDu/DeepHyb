# Set the value of theta
THETA=0.1
SITES=100000
NAME=10w

for NUM in $(seq 1 100)
do
    # Generate trees
    ms 13 $(echo "$SITES * 0.5" | bc) -T -I 5 3 3 3 3 1 -ej 0.5 3 1 -ej 0.5 2 1 -ej 1.0 4 1 -ej 1.5 5 1 | grep '^(' > trees_f1-$NUM.tre
    ms 13 $(echo "$SITES * 0.5" | bc) -T -I 5 3 3 3 3 1 -ej 0.5 3 4 -ej 0.5 2 1 -ej 1.0 4 1 -ej 1.5 5 1 | grep '^(' >> trees_f1-$NUM.tre

    # seq-gen simulation using GTR+I+G
    seq-gen -mGTR -s $THETA -l 1 -r 1.0 0.2 10.0 0.75 3.2 1.6 \
        -f 0.15 0.35 0.15 0.35 -i 0.2 -a 5.0 -g 3 -q < trees_f1-$NUM.tre > seqs_f1-$NUM.txt

    # Convert to HyDe format
    python3 seqgen2hyde.py seqs_f1-$NUM.txt data_$NAME_$NUM.phy data_$NAME_$NUM.imap


    sed -i '1,3d' data_$NAME_$NUM.phy
    python3 run_hyde.py -i data_$NAME_$NUM.phy -m map.imap -o out -n 10 -t 4 -s $SITES --prefix hyde_f1-$NUM

    rm -f trees_f1-$NUM.tre seqs_f1-$NUM.txt
done

