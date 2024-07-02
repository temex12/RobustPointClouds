@echo off
setlocal

REM Set the path to your mmdetection3d project directory
set MMDET3D_DIR=C:\Users\temex\Desktop\mmdet3dProj\mmdetection3d

REM Set paths to your log files
set BASELINE_LOG=C:\Users\temex\Desktop\mmdet3dProj\trainingOutputBaselineModel\20240308_003852\vis_data\20240308_003852
set ADVERSARIAL_LOG=C:\Users\temex\Desktop\mmdet3dProj\adversarialTrainingOutputW\20240316_213106\vis_data\20240316_213106

REM Change directory to mmdetection3d
cd %MMDET3D_DIR%

REM Plot the metrics, adjust --keys to your metrics in the logs
python tools\analysis_tools\analyze_logs.py plot_curve %BASELINE_LOG%.json %ADVERSARIAL_LOG%.json --keys loss_cls --out C:\Users\temex\Desktop\mmdet3dProj\metrics_comparison.pdf

endlocal
