#!/bin/bash
for m in BCV B C V; do
	python model_test.py --modality $m \
		--num_units 64 \
		--lr1 1e-5 \
		--lr2 1e-4 \
		--lr_decay 0.8 \
		--epoch 8 \
		--loss_weight 10 \
		--data_name data/audio_0426En \
		--is_diff True \
		--train_vgg True \
		--trained_layers 12 \
		--train_name cough_model
done
