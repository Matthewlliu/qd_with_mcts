CUDA_VISIBLE_DEVICES=4 python mcts_data_collection.py \
    --expand_width 3 \
    --save_path /data1/lhw/qd_mcts/reward_data_hard \
    --rollout_times 50 \
    --start_iter 0 \
    --save_llm_result_for_debug \
    --use_context