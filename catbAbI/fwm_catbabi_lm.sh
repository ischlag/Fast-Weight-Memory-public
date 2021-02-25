python3 main.py with p.dataset_name=catbAbI_v1_2 \
	p.model_name=lm_fwm \
	p.trainer_name=catbAbI_trainer \
	p.train_batch_size=32 \
	p.seed=1 \
	p.max_steps=50000 \
	p.learning_rate=0.001 \
    p.regularize=0.0 \
	p.ra_mode=False \
    p.residual=False \
	p.embedding_size=256 \
    p.hidden_size=256 \
	p.r_size=32 \
	p.t_size=32 \
	p.s_size=32 \
	p.n_reads=3 \
	p.run=""

