# import yt_dlp
# import sys  # Để nhận tham số từ dòng lệnh
# import os

# def download_videos(playlist_url, output_dir='./videos'):
#     # Tạo thư mục lưu video nếu chưa tồn tại
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Cấu hình yt_dlp
#     ydl_opts = {
#         'outtmpl': f'{output_dir}/%(title)s.%(ext)s',  # Lưu video với tên tiêu đề và định dạng tương ứng
#         'format': 'best',  # Chọn video có chất lượng tốt nhất
#         'noplaylist': False,  # Cho phép tải toàn bộ playlist
#     }

#     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#         ydl.download([playlist_url])

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python download_videos.py <playlist_url>")
#         sys.exit(1)
    
#     # Lấy URL playlist từ tham số dòng lệnh
#     playlist_url = sys.argv[1]
    
#     # Gọi hàm tải video
#     download_videos(playlist_url)
import os
import gdown

def download_videos(drive_folder_link, output_dir='./videos'):
    
    folder_id = drive_folder_link.split("/")[-1]

    os.makedirs(output_dir, exist_ok=True)
    
    # Tải toàn bộ file trong folder
    gdown.download_folder(id=folder_id, output=output_dir, quiet=False, use_cookies=False)
    print("Download completed!")
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python download_videos.py <drive_folder_link>")
        sys.exit(1)
    
    # Lấy URL playlist từ tham số dòng lệnh
    drive_folder_link = sys.argv[1]
    
    # Gọi hàm tải video
    download_videos(drive_folder_link)
