python run_sst2_classification.py \
--data_dir=/root/ysr/bert/bert/SST/SST2 \
--output_dir=/root/bert/bert/SST/output \
--vocab_file=/root/ysr/bert/bert/SST/uncased_L-4_H-256_A-4/vocab.txt \
--bert_config_file=/root/ysr/bert/bert/SST/uncased_L-4_H-256_A-4/bert_config.json \
--init_checkpoint=/root/ysr/bert/bert/SST/uncased_L-4_H-256_A-4/bert_model.ckpt \
--do_train=False \
--do_eval=False \
--do_predict=True \
--num_train_epochs=1.0