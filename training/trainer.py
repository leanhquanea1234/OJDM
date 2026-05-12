# Pray to the machine spririt. Thy lord may i ask for your blessin. im willing to offer my life's span for thy lord.
from ultralytics import YOLO

model = YOLO("OJ_model_v1.pt")

# model.train(data="datasets/OJ-v2/data.yaml")

results = model.train(
    data="datasets/OJ-v2/data.yaml",
    epochs=100,                  # 400枚の画像には100エポックが適切です
    imgsz=640,                   # 標準的な解像度
    batch=8,                     # CPU負荷を抑えるために小さめのバッチサイズ
    device="cpu",                # Intel HD GraphicsはCUDA非対応のためCPUを指定
    workers=4,                   # i5-7400Tの4コアに合わせる
    cache=True,                  # メモリ(RAM)に余裕があるためキャッシュを有効化して高速化
    optimizer="auto",            # 小規模データセットにはAdamWが自動選択されます
    patience=20,                 # 改善が見られない場合に早期終了する設定
    project="OJ_Detection",      # プロジェクト名
    name="OJ_model_v2"  # 実行名
)
"""
trained_model = YOLO("runs/detect/train/weights/best.pt")

results = trained_model("./people.jpg")
print(results[0].names)
"""

