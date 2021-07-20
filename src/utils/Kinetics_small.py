from pathlib import Path
from shutil import copy

rootdir = Path('.')
videos = list(rootdir.glob("**/*.mp4"))
classes = set()
for v in videos:
    v = str(v).split('/')
    classes.add(v[0])
classes = list(classes)

Path("train").mkdir(parents=True, exist_ok=True)
with open("train/train.csv", "w") as f:
    for v in videos:
        v_split = str(v).split('/')
        label = classes.index(v_split[0])
        copy_path = Path("train") / v_split[1]
        copy(v, copy_path)
        f.write("%s %d\n" % (str(copy_path.absolute()), label))
    
