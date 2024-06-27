@echo off
setlocal

set CONFIG_PATH=C:\Users\temex\Desktop\mmdet3dProj\configs\second_hv_secfpn_8xb6-80e_kitti-3d-car.py
set CHECKPOINT_PATH=C:\Users\temex\Desktop\mmdet3dProj\adversarialTrainingOutputW\epoch_40.pth
set ADVERSARIAL_PKL_PATH=C:\Users\temex\Desktop\mmdet3dProj\advTestOutput2n\adv_second_kitti_results\pred_instances_3d.pkl
set SHOW_DIR=C:\Users\temex\Desktop\mmdet3dProj\visResults

cd C:\Users\temex\Desktop\mmdet3dProj\mmdetection3d

REM Test the model and save the output to a file. This is optional if you already have the results file.
REM python tools/test.py %CONFIG_PATH% %CHECKPOINT_PATH% --out %ADVERSARIAL_PKL_PATH% --show --show-dir %SHOW_DIR%
set PYTHONPATH=C:\Users\temex\Desktop\mmdet3dProj\mmdetection3d;%PYTHONPATH%

REM Alternatively, visualize results from a previously saved pickle file
python tools/misc/visualize_results.py %CONFIG_PATH% --result %ADVERSARIAL_PKL_PATH% --show-dir %SHOW_DIR%

echo Visualization complete.
pause
