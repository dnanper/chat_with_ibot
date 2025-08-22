import cv2
import os

input_dir = '/home/dnanper/Downloads/dataset/videos'
output_dir = '/home/dnanper/Downloads/dataset/images_2'

FRAME_RATE = 10 # 10 pic / second

VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mkv', '.webm')

def extract_frames_from_folder():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Đã tạo thư mục chính: {output_dir}")
    try:
        files = os.listdir(input_dir)
    except FileNotFoundError:
        print(f" LỖI: Không tìm thấy thư mục '{input_dir}'. Vui lòng tạo thư mục và đặt video vào đó.")
        return

    for filename in files:
        if filename.lower().endswith(VIDEO_EXTENSIONS):
            video_path = os.path.join(input_dir, filename)
            video_name_without_ext = os.path.splitext(filename)[0]

            video_output_folder = output_dir
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"  Lỗi: Không thể mở file video {filename}")
                continue
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            if original_fps == 0:
                print(f"  Cảnh báo: Không đọc được FPS của video {filename}. Giả định là 30.")
                original_fps = 30
                
            frame_interval = int(original_fps / FRAME_RATE)
            if frame_interval == 0:
                frame_interval = 1

            frame_count = 0
            saved_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    image_name = f"frame_{saved_count:05d}.jpg"
                    save_path = os.path.join(video_output_folder, image_name)
                    cv2.imwrite(save_path, frame)
                    saved_count += 1
                
                frame_count += 1
            
            cap.release()
            print(f"  Hoàn thành! Đã lưu {saved_count} frame vào thư mục '{video_output_folder}'")

    print("\n Tất cả video đã được xử lý xong! ✨")
if __name__ == "__main__":
    extract_frames_from_folder()