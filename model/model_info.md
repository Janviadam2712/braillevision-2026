# Model Information

**Model:** YOLOv8n fine-tuned on Braille Detection V2

**Classes:** 26 (A-Z, Grade 1 Braille)

**Epochs:** 30 | **Image size:** 640x640

**Framework:** PyTorch (Ultralytics)

## Judge Access to Model Weights

Download [best.pt](http://best.pt) here:

**[https://drive.google.com/file/d/1AzDoFNU2fTah3yPAtFrOVRUPygqiULlB/view?usp=sharing](https://drive.google.com/file/d/1AzDoFNU2fTah3yPAtFrOVRUPygqiULlB/view?usp=sharing)**

**To load the model:**

from ultralytics import YOLO

model = YOLO('[best.pt](http://best.pt)')

print(model.names)

