import cv2
import numpy as np
import os

def is_image_mostly_black_simple(image_path, threshold=90):
    """Phiên bản đơn giản chỉ trả về True/False"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    
    black_pixels = np.sum(img <= 30)
    total_pixels = img.shape[0] * img.shape[1]
    black_percentage = (black_pixels / total_pixels) * 100
    
    return black_percentage > threshold

def split_image_auto(image_path, output_dir, debug=False):
    # Kiểm tra file
    if not os.path.exists(image_path):
        print(f"File không tồn tại: {image_path}")
        return
        
    img = cv2.imread(image_path)
    if img is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img.shape[:2]
    
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # 2. Canny với ngưỡng thấp cho đường kẻ mỏng
    edges = cv2.Canny(blurred, 20, 60, apertureSize=3)
    
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    lines = cv2.HoughLinesP(
        edges,
        rho=1,                          # Độ phân giải khoảng cách (pixel)
        theta=np.pi/180,                # Độ phân giải góc (radian)
        threshold=30,                   # GIẢM từ 100 xuống 30
        minLineLength=width * 0.3,      # GIẢM từ 0.8 xuống 0.3
        maxLineGap=50                   # TĂNG từ 10 lên 50
    )
    
    if debug:
        debug_img = img.copy()
        cv2.imwrite(f"debug_edges_{os.path.basename(image_path)}", edges)
        
    if lines is None:
        print("Không tìm thấy đường kẻ nào.")
        # Thử với tham số "desperate" (tuyệt vọng)
        print("Thử lại với tham số nhạy hơn...")
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=15,
            minLineLength=width * 0.2,
            maxLineGap=100
        )
        if lines is None:
            print("Vẫn không tìm thấy đường kẻ.")
            return
    
    split_points = [0]
    horizontal_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        angle_threshold = 10
        if abs(y1 - y2) <= angle_threshold:
            mid_y = (y1 + y2) // 2
            line_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            horizontal_lines.append({
                'y': mid_y,
                'length': line_length,
                'coords': (x1, y1, x2, y2)
            })
            
            if debug:
                cv2.line(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    horizontal_lines.sort(key=lambda x: x['length'], reverse=True)
    for line in horizontal_lines:
        y = line['y']
        if not any(abs(y - existing) < 20 for existing in split_points):
            split_points.append(y)
    
    split_points.append(height)
    split_points = sorted(split_points)
    
    if debug:
        cv2.imwrite(f"debug_lines_{os.path.basename(image_path)}", debug_img)
    
    print(f"Tìm thấy {len(split_points)-1} vùng cắt tại: {split_points}")
    
    for i in range(len(split_points) - 1):
        top = split_points[i]
        bottom = split_points[i + 1]
        
        if bottom - top < 300:
            continue
            
        cropped_img = img[top:bottom, :]
        cropped_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        black_pixels = np.sum(cropped_gray <= 30)
        total_pixels = cropped_gray.size
        black_percentage = (black_pixels / total_pixels) * 100
        print(black_percentage)
        if black_percentage > 95:
            # print(f"Vùng cắt {i+1} chủ yếu là màu đen.")
            continue

        base_name = os.path.basename(image_path).split('.')[0]
        output_path = os.path.join(output_dir, f"{base_name}_part_{i+1}.png")
        cv2.imwrite(output_path, cropped_img)
        print(f"Đã lưu: {output_path}")
        break

input_dir = '/home/dnanper/Downloads/dataset/images'
output_dir = '/home/dnanper/Downloads/dataset/images_processed'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(input_dir, filename)
        split_image_auto(image_path, output_dir,debug=False)
        print(f"Đã xử lý: {filename}")
        # break
