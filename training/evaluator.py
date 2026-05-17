import matplotlib
matplotlib.use('Agg')  # GUIエラー防止 (Prevent GUI backend error)

from ultralytics import YOLO

# 学習済みモデルのロード (Load your existing trained model)
# "best.pt" のパスは、実際の保存場所に書き換えてください
model = YOLO("runs/detect/OJ_Detection/OJ_model_v2-2/weights/best.pt")

# 検証のみを実行 (Run evaluation only)
metrics = model.val(
    data="datasets/OJ-v2/data.yaml",
    plots=True,          # グラフを生成する
    save_json=True,      # 詳細な結果を保存
    name="OJ_evaluation" # 結果を保存するディレクトリ名
)

print("Evaluation complete. Check 'runs/detect/OJ_evaluation/' for graphs.")
