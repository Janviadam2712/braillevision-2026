# Annotation Information

Format: YOLOv8 YOLO TXT

Each .txt file corresponds to an image with same filename.

Annotation format per line:

class_id x_center y_center width height

(all values normalized 0-1)

Example:

0 0.523 0.441 0.087 0.156

(class 0 = letter 'a', centered at 52.3% x, 44.1% y)

Full annotations available with dataset:

[https://universe.roboflow.com/braille-jjezl/braille-detection-v2-xpwue](https://universe.roboflow.com/braille-jjezl/braille-detection-v2-xpwue)

Class mapping:

0=a, 1=b, 2=c, 3=d, 4=e, 5=f, 6=g, 7=h, 8=i, 9=j,

10=k, 11=l, 12=m, 13=n, 14=o, 15=p, 16=q, 17=r, 18=s,

19=t, 20=u, 21=v, 22=w, 23=x, 24=y, 25=z