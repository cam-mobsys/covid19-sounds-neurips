#!/bin/bash
python model_test.py --modality BCV \
	--num_units 64 \
	--lr1 5e-5 \
	--lr2 1e-4 \
	--data_name data/audio_0426En \
	--is_diff True \
	--train_vgg True \
	--trained_layers 12 \
	--train_name covid_model
for m in B C V; do
	python model_test.py --modality $m \
		--num_units 64 \
		--lr1 5e-5 \
		--lr2 1.0e-4 \
		--loss_weight 10 \
		--data_name data/audio_0426En \
		--is_diff True \
		--train_vgg True \
		--trained_layers 8 \
		--train_name covid_model_single \
		--epoch 10
done
