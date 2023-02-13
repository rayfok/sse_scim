#!/bin/bash

### TEST SET ###
# NAACL 2022: 1 - 442 (442)
# ================================
# TOTAL: 443 papers

### TRAIN SET ###
# NAACL 2021: 1 - 477 (477)
# NAACL 2019: 1001 - 1424 (424)
# NAACL 2018: 1001 - 1205 (205)
# ACL 2022: 1 - 603 (603)
# ACL 2021: 1 - 571 (571)
# ACL 2020: 1 - 778 (778)
# ================================
# TOTAL: 3,058

while getopts "c:s:e:" opt; do
  case $opt in
    c) corpus="$OPTARG" ;;
    s) start="$OPTARG" ;;
    e) end="$OPTARG" ;;
  esac
done

for (( i=$start; i<=$end; i++ ))
  do

  if [[ $corpus == "NAACL22" ]]; then
    src="https://aclanthology.org/2022.naacl-main.$i.pdf"
  elif [[ $corpus == "NAACL21" ]]; then
    src="https://aclanthology.org/2021.naacl-main.$i.pdf"
  elif [[ $corpus == "NAACL19" ]]; then
    src="https://aclanthology.org/N19-$i.pdf"
  elif [[ $corpus == "NAACL18" ]]; then
    src="https://aclanthology.org/N18-$i.pdf"
  elif [[ $corpus == "ACL22" ]]; then
    src="https://aclanthology.org/2022.acl-long.$i.pdf"
  elif [[ $corpus == "ACL21" ]]; then
    src="https://aclanthology.org/2021.acl-long.$i.pdf"
  elif [[ $corpus == "ACL20" ]]; then
    src="https://aclanthology.org/2020.acl-main.$i.pdf"
  else
    echo "Invalid corpus:" $corpus
    exit 1
  fi

  python -m sse_skimming \
          src=$src \
          dst=./output
done
