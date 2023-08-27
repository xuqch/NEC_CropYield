#!/bin/sh
NAME='/home/xuqch3/anaconda3/bin/python -m joblib.externals.loky.backend.popen_loky_posix --process-name'
# NAME='ET_flux.py'
echo $NAME
ID=`ps -ef | grep ^xuqch3 | grep "$NAME" | grep -v "$0" | grep -v "grep" | awk '{print $2}'`
# ID=`ps -ef | grep ^xuqch3 | grep -v "$0" | grep -v "grep" | awk '{print $2}'`
# ps -ef | grep ^xuqch3 | grep "/home/xuqch3/anaconda3/bin/python -m joblib.externals.loky.backend.popen_loky_posix --process-name" |
echo "---------------"
for id in $ID;do
kill -9 $id
echo "killed $id"
done
echo "---------------"