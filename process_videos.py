import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
from concurrent.futures import ThreadPoolExecutor
import time  # Thêm thư viện time để delay giữa các thread

# Mediapipe Pose
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Load model đã huấn luyện
model = tf.keras.models.load_model("model_2_classes.h5")

# Class labels
class_labels = ["posing", "no posing"]

# Landmarks cần thiết
selected_landmarks = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
n_time_steps = 10

# Hàm xử lý landmarks
def make_landmark_timestep(results):
    c_lm = []
    for id in selected_landmarks:
        lm = results.pose_landmarks.landmark[id]
        c_lm.extend([lm.x, lm.y, lm.z])
    return c_lm

# Hàm dự đoán
def detect(model, lm_list):
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    results = model.predict(lm_list)
    predicted_class_index = np.argmax(results, axis=1)[0]
    predicted_prob = np.max(results, axis=1)[0]
    return class_labels[predicted_class_index] if predicted_prob >= 0.5 else "no posing"

# Hàm xử lý từng video
def process_video(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    lm_list = []
    frame_buffer = []
    warmup_frames = 60

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    while True:
        success, frame = cap.read()
        if not success:
            break

        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)

        if results.pose_landmarks:
            c_lm = make_landmark_timestep(results)
            lm_list.append(c_lm)
            if len(lm_list) == n_time_steps:
                label = detect(model, lm_list)
                lm_list.clear()
                if label == "posing":
                    frame_buffer.append(frame)

    if frame_buffer:
        output_file = os.path.join(output_dir, f"{base_name}_processed.mp4")
        height, width, _ = frame_buffer[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        for frame in frame_buffer:
            out.write(frame)
        out.release()
        print(f"Saved processed video: {output_file}")

    cap.release()

# Hàm xử lý tất cả video trong thư mục
def process_all_videos(input_dir, output_dir, num_threads=2, delay_between_threads=600):
    video_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.mp4', '.avi'))]
    os.makedirs(output_dir, exist_ok=True)

    # def delayed_process(video_path, output_dir):
    #     process_video(video_path, output_dir)
    #     time.sleep(delay_between_threads)

    # with ThreadPoolExecutor(max_workers=num_threads) as executor:
    #     futures = [executor.submit(delayed_process, video_path, output_dir) for video_path in video_files]
    #     for future in futures:
    #         future.result()
    for video_path in video_files:
        process_video(video_path, output_dir)

if __name__ == "__main__":
    input_dir = "./videos"  # Thư mục chứa video đã tải
    output_dir = "./processed_videos"  # Thư mục lưu video đã xử lý
    process_all_videos(input_dir, output_dir)
