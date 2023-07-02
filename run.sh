#!/bin/bash
echo "Start Optical calculation..."
echo ""
time ./optcri_loop_site.sh || exit 1 
echo "Done"
