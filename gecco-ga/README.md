## Giải thích

### Mã hóa

Các cá thể chứa các chromosome 2 lớp:
- lớp trên đại diện cho thứ tự các node sẽ được phục vụ
- lớp dưới đại diện cho mặt nạ để xác định chỉ định của node đối với thiết bị nào (0: xe tải, +1/-1: UAV)

### Cơ chế sửa chữa
1. Đầu tiên là phải đặt lại giá trị mặt nạ về 0 đối với các mã +1 và -1 vị phạm trọng tải UAV
2. Nếu có sự vi phạm trọng tải đối với môi lộ trình của tải và UAV thì sẽ căn bằng tải theo hàm balance
3. Khi có 1 chuỗi gene liên tiếp có giá trị mặt nạ thuộc {-1, 1} thì đặt lại mặt nạ ở các vị trí xen kẽ bằng 0
4. Các chuỗi gene của drone có các giá trị +1/-1 liên tiếp sẽ đại diên cho một trip của drone
  1. Các chuỗi gene kể từ gene đâu tiền khiến chuỗi gene vi phạm trọng tải của drone hoặc nằm ngoài phạm vi di chuyển của drone thì sẽ đổi dấu tất cả các mặt nạ {-1, 1}

5. Chia chromosome để xác định ra lộ trình của tải vói lộ trình của drone tương ứng
  1. Xác định điểm cất cánh, điểm hạ cánh của lộ trình của drone
  2. Lập lịch hành trình cho mỗi lộ trình được chia ra
