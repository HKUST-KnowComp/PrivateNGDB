CUDA_VISIBLE_DEVICES=2 python ../model/train_baseline_privacy.py \
    --train_query_dir ../input_files_privacy/YAGO15K_train_queries.pkl \
    --valid_query_dir ../input_files_privacy/YAGO15K_valid_queries.pkl \
    --test_query_dir ../input_files_privacy/YAGO15K_test_private_public_queries.pkl \
    --test_private_query_dir ../input_files_privacy/YAGO15K_private_1p_queries.json \
    --data_name YAGO15K \
    --model q2b \
    --batch_size 1024 \
    --log_steps 60000 \
    --mi \
    --privacy_multiplier 0.1
   
