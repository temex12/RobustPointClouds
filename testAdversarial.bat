@echo off
setlocal

cd C:\Users\temex\Desktop\mmdet3dProj\mmdetection3d

set CONFIG_PATH=C:\Users\temex\Desktop\mmdet3dProj\mmdetection3d\configs\second\second_hv_secfpn_8xb6-80e_kitti-3d-car.py
set CHECKPOINT_PATH=C:\Users\temex\Desktop\mmdet3dProj\adversarialTrainingOutput3\epoch_40.pth
set OUTPUT_DIR=C:\Users\temex\Desktop\mmdet3dProj\advTestOutput3new

python tools/test.py %CONFIG_PATH% %CHECKPOINT_PATH% --launcher none --cfg-options test_evaluator.pklfile_prefix="%OUTPUT_DIR%\adv_second_kitti_results" submission_prefix="%OUTPUT_DIR%\adv_second_kitti_results" --work-dir %OUTPUT_DIR%

