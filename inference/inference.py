# inference.py — BrailleBridge
# Run: python inference/inference.py --source IMAGE_PATH --weights best.pt

import argparse
from ultralytics import YOLO

def run_inference(source, weights="best.pt"):
    model = YOLO(weights)
    results = model(source)
    for result in results:
        print(f"Detected {len(result.boxes)} Braille dots")
        for box in result.boxes:
            label = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            print(f"  {label} — confidence: {conf:.2f}")
    output = "demo/screenshots/inference_output.jpg"
    results[0].save(filename=output)
    print(f"Saved to {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--weights", default="best.pt")
    args = parser.parse_args()
    run_inference(args.source, args.weights)