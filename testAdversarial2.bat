@echo off
setlocal

cd C:\Users\temex\Desktop\mmdet3dProj\mmdetection3d

set CONFIG_FILE=C:\Users\temex\Desktop\mmdet3dProj\configs\adversarial\adversarial-second_hv_secfpn_8xb6-80e_kitti-3d-car.py
set CHECKPOINT_PATH=C:\Users\temex\Desktop\mmdet3dProj\adversarialTrainingOutput1\epoch_40.pth
set OUTPUT_DIR=C:\Users\temex\Desktop\mmdet3dProj\advTestOutput2

python tools/test.py %CONFIG_FILE% %CHECKPOINT_PATH% --launcher none --cfg-options test_evaluator.pklfile_prefix="%OUTPUT_DIR%\adv_second_kitti_results" submission_prefix="%OUTPUT_DIR%\adv_second_kitti_results" --work-dir %OUTPUT_DIR%
