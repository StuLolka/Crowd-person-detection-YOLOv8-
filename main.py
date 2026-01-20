import cv2
from pathlib import Path
from ultralytics import YOLO
import subprocess
import imageio_ffmpeg


video_path = Path('src/crowd.mp4')
output_dir = Path('output')
output_dir.mkdir(exist_ok=True)

model = YOLO('yolov8n.pt')
params = {
    'conf': 0.4,
    'iou': 0.85,
    'imgsz': 1920
}

cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_dir = output_dir / 'frames'
frame_dir.mkdir(exist_ok=True)

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, **params, verbose=False)
    r = results[0]
    names = model.names

    for box in r.boxes:
        cls_id = int(box.cls.item())
        cls_name = names[cls_id]
        if cls_name != 'person':
            continue

        conf = float(box.conf.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'person: {conf:.2f}'
        cv2.putText(frame, label, (x1, max(0, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


    frame_path = frame_dir / f'frame_{frame_count:04d}.jpg'
    cv2.imwrite(str(frame_path), frame)
    frame_count += 1

cap.release()

out_path = output_dir / f'result.mp4'

ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

cmd = [
    ffmpeg_exe,
    '-y',
    '-framerate', str(fps),
    '-i', str(frame_dir / "frame_%04d.jpg"),
    '-c:v', 'libx264',
    '-pix_fmt', 'yuv420p',
    '-crf', '23',
    '-preset', 'medium',
    '-movflags', '+faststart',
    str(out_path)
]

result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode != 0:
    print(f'❌ Ошибка ffmpeg:\n{result.stderr}')
else:
    print(f'✅ Видео создано: {out_path}')

