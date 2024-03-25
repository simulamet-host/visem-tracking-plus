#!/bin/bash

# Create sorted lists of file names for both directories
find /home/pal.bentsen/D1/datasets2024/BBOX-2-OBB-v1-stockModels/segOutput -type f | sed 's#.*/##' | sort > /tmp/list1.txt
find /home/pal.bentsen/D1/datasets2024v2/tempOutputForPipeline/labels-segment -type f | sed 's#.*/##' | sort > /tmp/list2.txt

# Use diff to compare the file lists
diff /tmp/list1.txt /tmp/list2.txt
