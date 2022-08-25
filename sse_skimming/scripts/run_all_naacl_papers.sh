#!/bin/bash
START=$1
END=$2

for (( i=$START; i<=$END; i++ ))
do
    python -m sse_skimming \
    src="https://aclanthology.org/2022.naacl-main.$i.pdf" \
    dst=./output
done
