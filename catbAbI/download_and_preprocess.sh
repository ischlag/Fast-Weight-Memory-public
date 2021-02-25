wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz -P .tmp
tar xf .tmp/tasks_1-20_v1-2.tar.gz -C .tmp
python3 preprocess/process_catbAbI.py
rm -r .tmp