# Reproduce PTB and WT2 FWM experiments

The code is forked from Uber AI Lab's [differentiable-plasticity repo](https://github.com/uber-research/differentiable-plasticity/tree/master/awd-lstm-lm) which is itself forked from the [Salesforce Language model toolkit](https://github.com/Smerity/awd-lstm-lm). The code runs on pytorch 0.4.1 and we use conda to setup the environment accordingly.

1. Install conda 4.8.3: https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html

2. ```conda create -n pytorch04 python=3.6```

3. ```conda activate pytorch04```

4. ```conda install pytorch=0.4.1 cuda92 -c pytorch```

5. Train the model: ```python3 -u main.py --data data/penn --dropouti 0.5 --dropouth 0.3 --dropout 0.4 --dropoute 0.1 --wdrop 0.66 --seed 141 --epochs 900 --save PTB2.pt --model FWMRNNv2 --emsize 400 --nhid 1150 --nlayers 3 --optimizer adam --lr 0.001 --batch_size 20 --ssize 32 --rsize 32 --prefix ptb_model --asgdtime 765 --asgd_lr 2.0 --log_dir logs_ptb | tee -a logs_ptb/train_fwm_ptb_141.out```

6. Evaluate all partitions: ```python3 -u eval.py --data data/penn --dropouti 0.5 --dropouth 0.3 --drop
out 0.4 --dropoute 0.1 --wdrop 0.66 --seed 141 --epochs 900 --save PTB2.pt --model FWMRNNv2 --emsize 400 --nhid 1150 --nlayers 3 --optimizer adam --lr 0
.001 --batch_size 20 --ssize 32 --rsize 32 --prefix ptb_model --asgdtime 765 --asgd_lr 2.0 --log_dir logs_ptb | tee -a logs_ptb/eval_fwm_ptb_141.out```

7. Extract losses on test data: ```python3 -u extract_losses.py --data data/penn --dropouti 0.5 --dropouth 
0.3 --dropout 0.4 --dropoute 0.1 --wdrop 0.66 --seed 143 --epochs 900 --save PTB2.pt --model FWMRNNv2 --emsize 400 --nhid 1150 --nlayers 3 --optimizer a
dam --lr 0.001 --batch_size 20 --ssize 32 --rsize 32 --prefix ptb_model --asgdtime 765 --asgd_lr 2.0 --log_dir logs_ptb```

8. ```pip install jupyter seaborn```

9. Start a jupyter notebook server and checkout the two notebooks for a comparison of the uncertainty between an AWD-LSTM and our FWM.