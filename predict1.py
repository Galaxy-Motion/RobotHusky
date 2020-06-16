from tensorflow.keras.models import load_model
import cv2

model = load_model('first_try.h5')
#model.summary()

def prepare(path):
    img_size = 240
    img_array = cv2.imread(path)
    new_array = cv2.resize(img_array, (img_size, img_size))
    return new_array.reshape(-1, img_size, img_size, 3)
num = 0
for i in range(0,147):
    input_im = prepare('test/grass/'+str(i)+'.jpg')
    pre_y = model.predict(input_im)
    if pre_y <0.5:
        num = num + 1

print(num/147)
