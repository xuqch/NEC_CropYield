python -u box.py TAGP > box.log 2>&1 &
#python -u box.py TWSO > box.log 2>&1 &
python -u density.py TAGP > density.log 2>&1 &
#python -u density.py TWSO > density.log 2>&1 &

echo 'done'