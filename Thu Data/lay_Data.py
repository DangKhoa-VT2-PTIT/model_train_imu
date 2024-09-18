import socket
import csv

HOST = '192.168.1.139' 
PORT = 8090

csv_file_path = 'Stop_Breathing.csv' 
print(f"File sẽ được lưu tại: {csv_file_path}")


#Hàm tách chuỗi nhận được từ cilent
def split_string_to_list(string):
    # Tách chuỗi thành các phần tử dựa trên dấu #
    parts = string.split("#")

    # Tạo một danh sách để lưu trữ kết quả
    result = []

    # Duyệt qua từng phần tử trong danh sách các phần tử
    for part in parts:
        # Tách mỗi phần tử thành một list con dựa trên dấu ,
        sub_list = part.split(",")
        
        # Chuyển đổi các phần tử từ kiểu str sang kiểu float
        sub_list = [float(item) for item in sub_list]

        # Thêm list con vào danh sách kết quả
        result.append(sub_list)

    return result


# Hàm để xử lý kết nối từ một client
def handle_connection(conn, addr):
    print(f"Connected by {addr}")

    sampleData = '' #Lưu data nhận được từ cilent

    #Lấy data từ cilent
    while True:
        data = conn.recv(10240000)
        if not data:
            break

        decoded_data = data.decode('utf-8')
        sampleData += decoded_data
        
    print("sampleData: " + sampleData +"\n")
    data_list = split_string_to_list(sampleData.strip(",").strip("#"));

    # Mở tệp CSV để ghi
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        print("Start write data to .csv\n")

        writer.writerow(['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ'])
        print("Start write data to .csv..\n")

        writer.writerows(data_list)
        print("Start write data to .csv...\n")

    print("Dữ liệu đã được lưu vào tệp "+csv_file_path+ " thành công.")

    conn.close()

# Bắt đầu server và tiếp tục lắng nghe các kết nối mới
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"Listening on {HOST}:{PORT}")

    while True:
        conn, addr = s.accept()
        handle_connection(conn, addr)
