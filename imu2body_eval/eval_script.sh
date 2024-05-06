# generate eval files 
python3 eval_by_files.py --data-config-path="./data_config/" --base-dir="/data/AMASS_smplh" --save-path="/data/imu2body_eval/eval_input_novel_1023/"    
# hps
python3 eval_by_files.py --data-config-path="./data_config/" --base-dir="/data/HPS" --save-path="/data/imu2body_eval/eval_input_novel_HPS_heightfix" --data-typ
e="hps"

# eval files on tc
python3 eval_by_files.py --data-config-path="./data_config/" --base-dir="/data/AMASS_smplh" --save-path="/data/imu2body_eval/eval_input_novel_TC_real/" --data-type="tc"    

# run model evaluation
python3 run_eval.py --test_name=eval_novel_root_sb_da --mode=test --config=amass_eval --eval-path="/data/imu2body_eval/eval_input_novel_1023/"  
# run imu eval
python3 run_eval.py --test_name=amass_imu_eval --mode=test --config=amass_imu_eval --eval-path="/data/imu2body_eval/eval_input_novel_TC_real/"  


# cam ready reproduce
# comment out height adjust related part
python3 eval_by_files.py --data-config-path="./data_config/" --base-dir="/data/AMASS_smplh" --save-path="/data/imu2body_eval/eval_input_novel_1023_heightfix/"  --data-type="amass"  
python3 eval_by_files.py --data-config-path="./data_config/" --base-dir="/data/HPS" --save-path="/data/imu2body_eval/eval_input_novel_HPS_heightfix/" --data-type="hps"
python3 eval_by_files.py --data-config-path="./data_config/" --base-dir="/data/AMASS_smplh" --save-path="/data/imu2body_eval/eval_input_novel_TC_real_heightfix/" --data-type="tc"

python3 run_eval.py --test_name=amass_imu_eval --mode=test --config=amass_imu_eval --eval-path="/data/imu2body_eval/eval_input_novel_TC_real_heightfix/"  
python3 run_eval.py --test_name=eval_novel_root_sb_da_long --mode=test --config=amass_eval --eval-path="/data/imu2body_eval/eval_input_novel_1023_heightfix/"  
python3 run_eval.py --test_name=eval_novel_root_sb_da_long --mode=test --config=amass_eval --eval-path="/data/imu2body_eval/eval_input_novel_HPS_heightfix/"  
