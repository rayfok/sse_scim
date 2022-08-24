#!/bin/bash

for i in {$0..$1}
do
    python -m sse_skimming \
        src="https://aclanthology.org/2022.naacl-main.$i.pdf" \
        dst=./output
done
