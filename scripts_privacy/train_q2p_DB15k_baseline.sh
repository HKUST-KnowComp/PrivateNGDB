CUDA_VISIBLE_DEVICES=1 python ../model/train_baseline.py \
    --train_query_dir ../input_files_privacy/DB15K_train_queries.pkl \
    --valid_query_dir ../input_files_privacy/DB15K_valid_queries.pkl \
    --test_query_dir ../input_files_privacy/DB15K_test_private_public_queries.pkl \
    --test_private_query_dir ../input_files_privacy/DB15K_private_1p_queries.json \
    --data_name DB15K \
    --model q2p \
    --batch_size 1024 \
    --log_steps 120000 
   