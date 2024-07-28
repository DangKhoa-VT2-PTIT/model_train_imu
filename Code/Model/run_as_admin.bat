@echo off
set script_path="C:\Users\ADMIN\Desktop\model_train_imu\DoAn_He_Thong_Nhung\Code\Model\Train_Model.py"
echo Running %script_path% with admin privileges.
powershell -Command "Start-Process python -ArgumentList '%script_path%' -Verb RunAs"
pause
