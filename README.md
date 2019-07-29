## Machine learning based Cat vs Dog
Algorithm predicts defined image "is dog or cat?" using Tensorflow backend.
I used Kaggle Cat vs Dog dataset but you're free to use your own dataset. If you will use your own dataset, you need to edit train.py.
If you will use Kaggle's dataset, set train and test dirs as "currentdir/train" and "currentdir/test".

## Usage
First, you need to install needed libraries.

For first time:
```python
python train.py -o normal
```
For training existing model:
```python
python train.py -o train
```
For classificate an image:
```python
python classification.py -i imagename
```
