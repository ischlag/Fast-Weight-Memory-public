EDIT: Source is Zihang Dai's tweet https://twitter.com/ZihangDai/status/1245905407350112256?s=19 which links to http://zihangdai.github.io/misc/ptb.zip


```bash
bash get_data.sh
python train.py --cuda
# see example_log.txt for an old run
```

- Due to the super limited size, in my opinion, PTB/WT2 are "regularization games" that should **NOT** be used to measure the **capacity** or **expressiveness** of models any more in the future. 
- Due to the reason mentioned above, the code includes various regularization tricks that may never generalize to other problems. Hence, the code will **NOT** be maintained in any way.
- Have fun with it if you are really interested in regularization and be clear about it when writing a paper.

