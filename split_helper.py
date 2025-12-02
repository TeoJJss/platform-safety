import os, random, shutil

images = "dataset/images"
labels = "dataset/labels"

train_imgs = "dataset/images/train"
val_imgs = "dataset/images/val"
train_lbls = "dataset/labels/train"
val_lbls = "dataset/labels/val"

# create folders if they don't exist
for d in [train_imgs, val_imgs, train_lbls, val_lbls]:
    os.makedirs(d, exist_ok=True)

# include multiple image extensions
valid_exts = [".jpg", ".jpeg", ".png"]
files = [f for f in os.listdir(images) if os.path.splitext(f)[1].lower() in valid_exts]

random.shuffle(files)
split = int(0.8 * len(files))

train_files = files[:split]
val_files = files[split:]

# copy train files
for f in train_files:
    shutil.copy(os.path.join(images, f), os.path.join(train_imgs, f))
    lbl = os.path.splitext(f)[0] + ".txt"
    if os.path.exists(os.path.join(labels, lbl)):
        shutil.copy(os.path.join(labels, lbl), os.path.join(train_lbls, lbl))

# copy val files
for f in val_files:
    shutil.copy(os.path.join(images, f), os.path.join(val_imgs, f))
    lbl = os.path.splitext(f)[0] + ".txt"
    if os.path.exists(os.path.join(labels, lbl)):
        shutil.copy(os.path.join(labels, lbl), os.path.join(val_lbls, lbl))
