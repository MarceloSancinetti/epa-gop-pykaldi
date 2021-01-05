#!/bin/bash -e



annotation="annotations_1"
output="logs"


rm -rf */labels/*
find . -name 'labels' -type d -delete

rm -rf $output
mkdir -p $output

python3 assign_reference.py  \
    --transcription-file reference_transcriptions.txt  \
    --annotation-dir $annotation \
    --output-dir $output >> $output/log.txt
