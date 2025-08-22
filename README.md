# Chatbot-MyAI

1. Thuật toán nhận dạng đường biên Canny (Canny Edge Detection):  
   Bước 1: Sử dụng Gaussian Blur -> Giảm nhiễu, làm mờ mọi pixel trong ảnh để tránh trường hợp tồn tại các khối pixels với Gradient quá lớn gây nhiễu trong việc phát hiện Edge. Tuy điều này cũng làm cho các đường biên mờ đi, nhưng sẽ không mất hẳn do chúng bản chất là vùng có thay đổi ánh sáng lớn.  
   Bước 2: Tính Gradient (Sobel) -> Tìm chỗ có Gradient thay đổi mạnh.  
   Bước 3: Non-Maximum Suppression (NMS): loại bỏ các biên dày, chỉ giữ lại biên mảnh nhất. Nguyên lý là so sánh pixel với các pixel xung quanh theo hướng graident. Nếu như nó không phải là lớn nhất thì loại bỏ -> Tránh việc nhầm lẫn biên với các vùng tối.  
   Bước 4: Double Threshold → lọc biên mạnh, yếu bằng các Threshold (strong / weak) -> phân loại pixel thành biên mạnh, biên yếu, hoặc không phải biên.
   Bước 5: Edge Tracking bằng thuật Hysteresis: Biên yếu có thể là biên thật (nếu nó nối tiếp biên mạnh), hoặc chỉ là nhiễu. Thuật toán Hysteresis sẽ loại bỏ các biên yếu đứng một mình (coi nó là nhiễu) và chỉ để lại những biên yếu nối với biên mạnh.

  1.1 Bổ sung về Gaussian Blur.  
    Gaussian Blur là 1 công thức, áp dụng cho từng pixel. Giá trị của các pixel sẽ được tính lại dựa theo tọa độ của nó, theo phân phối chuẩn Gauss (ở giữa cao, càng ra ngoài càng thấp - chuông 2D). Điều này làm mờ đi ảnh đầu vào.  
    Trong convolutional network, cũng tồn tại các ma trận kernel (filter/mask) ở các layer. Các kernel này sẽ trượt qua input và tạo ra output đặc biệt phụ thuộc vào giá trị của kernel đó. Có thể là làm mờ , làm nét, tách hướng, ....  
    Do đó, ta có thể sử dụng kĩ thuật này để làm mờ ảnh, thay vì Gaussian Blur.
