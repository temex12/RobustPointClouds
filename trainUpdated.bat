@echo off

set CONFIG_FILE=C:\Users\temex\Desktop\mmdet3dProj\configs\adversarial\adversarial-second_hv_secfpn_8xb6-80e_kitti-3d-car.py
set WORK_DIR=C:\Users\temex\Desktop\mmdet3dProj\NewAdvTrainingOutput
set CHECKPOINT_FILE=C:\Users\temex\Desktop\mmdet3dProj\checkpoints\second_hv_secfpn_8xb6-80e_kitti-3d-car-75d9305e.pth

set PYTHONPATH=%PYTHONPATH%;C:\Users\temex\Desktop\mmdet3dProj

python train.py %CONFIG_FILE% --work-dir %WORK_DIR% --cfg-options load_from="%CHECKPOINT_FILE%" --auto-scale-lr --amp


