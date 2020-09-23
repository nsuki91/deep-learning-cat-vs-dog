## WARNING: App doesn't work with Tensorflow 2.x | I recommend 1.14.0
## Deep learning based Cat vs Dog
The algorithm predicts the defined image "is a dog or cat?" using Tensorflow backend. 
I used Kaggle Cat vs Dog dataset but you're free to use your own dataset. If you will use your own dataset, you need to edit train.py.
If you will use Kaggle's dataset, set train and test dirs as "currentdir/train" and "currentdir/test".

![alt text](https://i.gyazo.com/4a3af617aa6f0f34591cbe8c519b264a.gif)
## Usage
First, you need to install the needed libraries.
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

For the first time:
```python
python train.py -o normal
```
For training the existing model:
```python
python train.py -o train
```
For classificate images:
```python
python gui.py
```
or just double click the gui.py and select the image from the file dialog.
