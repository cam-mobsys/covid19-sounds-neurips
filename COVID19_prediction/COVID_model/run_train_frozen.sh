#!/bin/bash
python model_train_vad.py --modality BCV \
	--num_units 64 \
	--lr1 5e-5 \
	--lr2 1e-4 \
	--loss_weight 10 \
	--data_name data/audio_0426En \
	--train_name covid_model_frozen
for m in B C V; do
	python model_train_vad.py --modality $m \
		--num_units 64 \
		--lr1 5e-5 \
		--lr2 1.0e-4 \
		--loss_weight 10 \
		--data_name data/audio_0426En \
		--train_name covid_model_single_frozen \
		--epoch 20
done
