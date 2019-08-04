## Machine learning based Cat vs Dog
Algorithm predicts defined image "is dog or cat?" using Tensorflow backend.
I used Kaggle Cat vs Dog dataset but you're free to use your own dataset. If you will use your own dataset, you need to edit train.py.
If you will use Kaggle's dataset, set train and test dirs as "currentdir/train" and "currentdir/test".

![alt text](https://i.gyazo.com/4a3af617aa6f0f34591cbe8c519b264a.gif)
## Usage
First, you need to install needed libraries.
```
tkinter
Pillow
cv2
numpy
tqdm
tflearn
Tensorflow
matplotlib
argparse
```

For first time:
```python
python train.py -o normal
```
For training existing model:
```python
python train.py -o train
```
For classificate images:
```python
python gui.py
```
or just double click the gui.py and select image from file dialog.
