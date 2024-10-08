Trong phần code này tổng sẽ có 4 folder:
- Arduino: Chứa tất cả code Arduino của đồ án :
	+ Lay_data: File chứa code thu data(Mục đích để train) và gửi lên server
	+ Nhung_model_esp32_RTOS - Web: 
		* gesture_model: File model sau khi train được chuyển vào đây để nhúng vào chip
		* Nhung_model_esp32_RTOS: File code Arduino, file này chứa code để nhúng vào chip ESP32. Bao gồm thu data để dự đoán, dự đoán nhãn từ data thu 			được, và gửi nhãn đó lên web
- Model: Chứa tất cả code, data dùng để train và model thu được sau khi train
	+ data_train: Chứa tất cả file .csv dùng để train model
	+ gesture_model.tflite:file thu được sau khi train model( file này chưa nhúng vào chip được mà cần thêm hàm chuyển đổi trong file Train_Model để nhúng 		vào chip)
	+ gesture_model: File này đã chuyển về C để nhúng vào chip. Được tạo cùng lúc với train model
	+ Train_Model: Dùng để tạo model nhúng vào chip
- Thu Data: 
	+ lay_Data: Chứa code server localhost dùng để nhận data lúc chip gửi data lên và lưu vào file .csv (bật file này trước khi chip gửi để nhận data) 
- Web: 
	+ nhung: File web đơn giản dùng nhận các nhãn được chip gửi đến và hiển thị nó lên web theo realtime 

"# model_train_imu" 
"# model_train_imu" 
