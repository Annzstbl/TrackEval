# <!-- {
#     "name": "评测",
#     "type": "debugpy",
#     "request": "launch",
#     "cwd":"${workspaceFolder}/TrackEval",
#     "program": "scripts/run_hsmot_rgb.py",
#     "args": [
#         "--USE_PARALLEL", "False",
#         "--METRICS", "HOTA","CLEAR","Identity",
#         "--TRACKERS_TO_EVAL", "motr/e2e_motr_r50_train_hsmot_rgb_23_l1_mmrotate_interval3",
#         "--TRACKER_SUB_FOLDER", "preds",
#         // "--TRACKERS_TO_EVAL", "motrv2/e2e_motr_r50_train_hsmot_rgb_1",
#         // "--TRACKER_SUB_FOLDER", "submit"
#         // "--TRACKERS_TO_EVAL", "yolo11/predict/hsmot_rgb_v11lobb_pretrainedweight9_predict_track",
#         // "--TRACKER_SUB_FOLDER", "preds"
#     ],
#     "justMyCode": false,
#     "env": {
#         "CUDA_VISIBLE_DEVICES": "0"
#     }

# } -->
# 第一个输入参数给Trackers
# 例如 motr/e2e_motr_r50_train_hsmot_rgb_23_l1_mmrotate_interval3 
TRACKERS_TO_EVAL=$1
TRACKERS_SUB_FOLDER=$2

cd /data/users/litianhao/hsmot_code/TrackEval/
python scripts/run_hsmot_rgb.py --USE_PARALLEL False --METRICS HOTA CLEAR Identity --TRACKERS_TO_EVAL ${TRACKERS_TO_EVAL} --TRACKER_SUB_FOLDER ${TRACKERS_SUB_FOLDER}