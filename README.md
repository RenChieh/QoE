# python code

Status: Not Started
Assign: renjie
Due: January 23, 2023
skill: python
狀態: Not started

## File

- download data from gc storage

```python

# 從 GCS 值區下載資料
#先建立好log與pic資料夾
!mkdir ./
!mkdir ./train
#圖檔放在資料夾中（來源：目的）
!gsutil -m cp -r gs://qoestorage/vm1/cnn.zip ./
```

- 解壓縮

```python
#unzip train data
!pip install zipfile
import os
import zipfile
#解壓縮要放到哪個目錄
url = '/content'
# zipfile example
def zip_list(file_path):
  zf = zipfile.ZipFile(file_path, 'r')
  zf.extractall(url)
 
if __name__ == '__main__':
  #壓縮檔位置
  file_path = '/content/cnn.zip'
  zip_list(file_path)
```

- 改檔名[https://ithelp.ithome.com.tw/m/articles/10273886](https://ithelp.ithome.com.tw/m/articles/10273886)

```python
#標記gaming=0
import os
#source
y=os.listdir('/content/CNN_training/0')
g=len(y)
#from to
for i in range(0, g):
   os.rename(f'/content/CNN_training/0/{y[i]}',f'/content/CNN_training/0/0_{i}.png')
```

- 改圖片格式

```python
#修改圖片格式
import os,sys
folder = '/content/test_vm2'
for filename in os.listdir(folder):
       infilename = os.path.join(folder,filename)
       if not os.path.isfile(infilename): continue
       oldbase = os.path.splitext(filename)
       newname = infilename.replace('png', 'jpg')
       output = os.rename(infilename, newname)
```

- 移動Ａ資料夾裡的檔案到Ｂ資料夾

```python
#移動目的
import os
 
file_source = '/content/CNN_training/1/'
file_destination = '/content/train/'
 
get_files = os.listdir(file_source)
 
for g in get_files:
    os.replace(file_source + g, file_destination + g)
```

- 列出資料夾裡面的檔案

```python
filenames = os.listdir("/content/train")
```

- 根據檔名“＿”前面的數字加標籤0 and 1，變成data frame

```python
categories = []
for filename in filenames:
    category = filename.split('_')[0]
    if category == '1':
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})
```

- 找資料夾裡面是否有讀取錯誤的檔案，並刪除
- [https://blog.csdn.net/qq_44936246/article/details/117962404](https://blog.csdn.net/qq_44936246/article/details/117962404)

```python
import os
import shutil
import warnings
import cv2
import io
from PIL import Image
warnings.filterwarnings("error",category=UserWarning)
base_dir="/content/cnn/test_CNN/0"
i=0
def is_read_successfully(file):
  try:
    imgFile =Image.open(file)
    return True
  except Exception:
    return False
for parent, dirs, files in os.walk(base_dir):
  for file in files:
    if not is_read_successfully(os.path.join(parent, file)):
      print(os.path.join(parent, file))
      #os.remove(os.path.join(parent, file))先確定程式可以跑在用這行刪除
      i=i+1
print(i)
```

## AI model

### train CNN model

```python
import numpy as np
import pandas as pd 
from tensorflow.keras.utils import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
```

```python
FAST_RUN = False
IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3
```

```python
filenames = os.listdir("/content/train")
categories = []
for filename in filenames:
    category = filename.split('_')[0]
    if category == 'gaming':
        categories.append(0)
    else:
        categories.append(1)

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})
```

```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# model.add(Dense(128, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid')) # 2 because we have cat and dog classes

#model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
```

```python
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
earlystop = EarlyStopping(patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
callbacks = [earlystop, learning_rate_reduction]
df["category"] = df["category"].replace({0: 'gaming', 1: 'not gameing'}) 
train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size=15
#Traning Generator
train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "/content/train/", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='binary',
    batch_size=batch_size
)
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "/content/train/", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='binary',
    batch_size=batch_size
)
print(validation_generator.class_indices)
print(validation_generator.n)
print(len(validation_generator))
X_val, y_val = validation_generator.__next__()
print(X_val.shape, y_val.shape)
X_list = []
y_list = []
for i in range(validation_generator.__len__()):
    X, y = validation_generator.__getitem__(i)
    X_list.append(X)
    y_list.append(y)

X_val = np.concatenate(X_list, axis=0)
y_val1 = np.concatenate(y_list, axis=0)
y_val_argmax = np.argmax(y_val1, axis=1).astype('uint8')
X_val.shape, y_val1.shape, y_val_argmax.shape, y_val_argmax[:10]
X_list[0].shape
example_df = train_df.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(
    example_df, 
    "/content/train/", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical'
)
plt.figure(figsize=(12, 12))
for i in range(0, 15):
    plt.subplot(5, 3, i+1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()
```

```python
import tensorflow as tf
model.reset_states()
model.summary()
epochs=3 if FAST_RUN else 10
history = model.fit(
        train_generator, 
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=total_validate//batch_size,
        steps_per_epoch=total_train//batch_size,
        callbacks=callbacks
  )
```

```python
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, epochs, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))

ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, epochs, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()
```

```python
model.save("model.h5")
```

```python
#標記gaming=0
import os
#source
y=os.listdir('/content/test')
g=len(y)
#from to
for i in range(0, g):
   os.rename(f'/content/test/{y[i]}',f'/content/test/test_{i}.jpg')
```

```python
test_filenames = os.listdir("/content/test/")
test_df = pd.DataFrame({
    'filename': test_filenames
})
nb_samples = test_df.shape[0]
```

```python
test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df, 
    "/content/test/", 
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    shuffle=False
)
```

```python
preds = model.predict(validation_generator, 
                                  steps=total_validate//batch_size + 1,
                                  verbose=1)
```

```python
from sklearn.metrics import classification_report
#(y_true, y_pred, *, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division='warn')[source]¶
target_names = ['gaming', 'nogaming']
y_true = y_val_argmax
y_pred = y_val_pred

print(classification_report(y_true, y_pred, target_names=target_names))

```

### Model prediction(.png)

- 匯入套件

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
model = load_model('my_model.h5')
```

- 跑單一個圖片的預測

```python
#跑單一一個檔
img = image.load_img("/content/cnn/test_CNN/1/Engine_Evolution_2022_(1871990)_11-03-22_04-15-00_Screenshot.png",target_size=(64,64))
x = image.img_to_array(img)
x = np.expand_dims(x,axis=0)
img_data = preprocess_input(x)
classes = model.predict(img_data)

classes
```

- 預測資料夾裡面所有的圖檔

```python
#read all pic 
import os
df_total = pd.DataFrame()
count = 0
z = os.listdir(r"/content/cnn/test_CNN/1")
for i in z:
  try:
    img = image.load_img(f"/content/cnn/test_CNN/1/{i}",target_size=(64,64))
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    img_data = preprocess_input(x)
    classes = model.predict(img_data) 
    max = np.max(classes[0])
    max_index = np.where(classes == max)
    print(f"{i}-finished")
    temp_dict = {"file":i,"max":max,"max_index":max_index[1],"0":classes[0][0],"1":classes[0][1]}
    df_temp = pd.DataFrame(temp_dict,index = [0])
    df_total = df_total.append(df_temp)
  
  except:
    print(f"error with the file-{i}")
    pass
```

### Model prediction(.jpg)

## file

```python
# 從 GCS 值區下載資料
#先建立好log與pic資料夾
!mkdir ./test_vm3

#圖檔放在資料夾中（來源：目的）
!gsutil -m cp -r gs://qoestorage/vm3/vm3_png.zip ./
```

- unzip data

```python
#unzip train data
!pip install zipfile
import os
import zipfile
#解壓縮要放到哪個目錄
url = '/content/test_vm3'
# zipfile example
def zip_list(file_path):
  zf = zipfile.ZipFile(file_path, 'r')
  zf.extractall(url)
 
if __name__ == '__main__':
  #壓縮檔位置
  file_path = '/content/vm3_png.zip'
  zip_list(file_path)
```

- check error image

```python
import os
import shutil
import warnings
import cv2
import io
from PIL import Image
warnings.filterwarnings("error",category=UserWarning)
base_dir="/content/test_vm3"
i=0
def is_read_successfully(file):
  try:
    imgFile =Image.open(file)
    return True
  except Exception:
    return False
for parent, dirs, files in os.walk(base_dir):
  for file in files:
    if not is_read_successfully(os.path.join(parent, file)):
      print(os.path.join(parent, file))
      os.remove(os.path.join(parent, file))
      i=i+1
print(i)
```

- 修改圖片格式

```python
#修改圖片格式
import os,sys
folder = '/content/test_vm3'
for filename in os.listdir(folder):
       infilename = os.path.join(folder,filename)
       if not os.path.isfile(infilename): continue
       oldbase = os.path.splitext(filename)
       newname = infilename.replace('png', 'jpg') #(old,new)
       output = os.rename(infilename, newname)
```

- image name store in the dataframe

```python
test_filenames = os.listdir("/content/test_vm3")
test_df = pd.DataFrame({
    'file': test_filenames
})
nb_samples = test_df.shape[0]
```

## model

- load model

```python
import numpy as np
import pandas as pd 
from tensorflow.keras.utils import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
from tensorflow.keras.models import load_model
filepath='/content/model.h5'
model = load_model(filepath)
```

- 圖片格式

```python
FAST_RUN = False
IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3
batch_size=15
```

- test generator

```python
test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df, 
    "/content/test_vm3/", 
    x_col='file',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    shuffle=False
)
```

- predict

```python
predict = model.predict(test_generator, steps=np.ceil(nb_samples/batch_size),
                                 verbose=1)
```

- 取最大值

```python
test_df['category'] = np.argmax(predict, axis=-1)
```

- bar plot

```python
test_df['category'].value_counts().plot.bar()
```

- see predict result

```python
sample_test = test_df.head(18)
sample_test.head()
plt.figure(figsize=(24, 24))
for index, row in sample_test.iterrows():
    filename = row['file']
    category = row['category']
    img = load_img("/content/test_vm3/"+filename, target_size=IMAGE_SIZE)
    plt.subplot(6, 3, index+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')' )
plt.tight_layout()
plt.show()
```

- 修改欄位名稱

```python
cnn_f3 = cnn_f3.rename(columns={'filename': 'file'}) #修改欄位名稱
```

## .txt preprocessing

- 用.切欄位取第0個值
- index重新排序

```python
x['file'] = x['file'].str.split('_StreamVideoTrace').str[0]
x=x.reset_index(drop=True)
```

- try-except

```python
x = pd.DataFrame()#create dataframe
b = os.listdir(r"/content/qoestorage/vm3/log")#list the file name
#sort the file name
b.sort(key = lambda x:x.split('.')[0][16:50].replace('_','').replace('(','').replace(')','').replace('-',''))

for i in b:
  try:
    df = network(f"{i}") #network is the function
    df["file"] = i  #filename store in file col
    x = x.append(df) #new data
  except:
    print(f"error with the file-{i}")
    pass
```

- create network function for deal with log.txt

```python
! pip install nums_from_string

#算每一個log參數的AVG,STD(修正版)
#num test
#使用 get_nums() 函數
# 載入現成套件

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import nums_from_string as nfs

def network(file):
  Path = "/content/qoestorage/vm3/log/"
  df = pd.read_csv(Path + file ,delimiter="\t",names=['data']) #將資料變成dataframe
  # Path = "/content/"
  # df = pd.read_csv(Path + file,delimiter="\t",names=['data']) #將資料變成dataframe
  df[['ping', 'server', 'client','link','packetloss']] = df.data.str.split(',', expand=True)

  #把data丟掉
  data2 = df.drop(labels=['data'],axis='columns')
  #顯示出NETWORK開頭的資料
  net_sta2=data2[data2.ping.str.startswith('NETWORK')] 

  net_sta2.reset_index(inplace=True, drop=True)

  ping_list = []
  server_list = []
  client_list = []
  link_list = []
  packetloss_list = []
  for i in range(len(net_sta2)):
    ping = net_sta2.iloc[i, 0]
    ping_new = nfs.get_nums(ping)
    #多個list轉在一起
    ping_num = [str(integer) for integer in ping_new]
    ping_nums = float("".join(ping_num))
    ping_list.append(ping_nums)

    server = net_sta2.iloc[i, 1]
    server_new = nfs.get_nums(server)
    server_num = [str(integer) for integer in server_new]
    server_nums = float("".join(server_num))
    server_list.append(server_nums)

    client = net_sta2.iloc[i, 2]
    client_new = nfs.get_nums(client)
    client_num = [str(integer) for integer in client_new]
    client_nums = float("".join(client_num))
    client_list.append(client_nums)

    link = net_sta2.iloc[i, 3]
    link_new = nfs.get_nums(link)
    link_num = [str(integer) for integer in link_new]
    link_nums = float("".join(link_num))
    link_list.append(link_nums)

    packetloss = net_sta2.iloc[i, 4]
    packetloss_new = nfs.get_nums(packetloss)
    packetloss_num = [str(integer) for integer in packetloss_new]
    packetloss_nums = float("".join(packetloss_num))
    packetloss_list.append(packetloss_nums)

  net_sta2.insert(5, 'ping_new', ping_list)
  net_sta2.insert(6, 'server_new', server_list)
  net_sta2.insert(7, 'client_new', client_list)
  net_sta2.insert(8, 'link_new', link_list)
  net_sta2.insert(9, 'packetloss_new', packetloss_list)
  net= net_sta2.drop(labels=['ping','server','client','link','packetloss'],axis='columns')

  #all frame data
  #抓frameage的資料
  frameage = pd.DataFrame(columns = ["frameage", "framesize", "cap_time", "con_time","Encode_time", "transfer_time", "Decode_time","Upload_time", "complete_time"])
  total = "total" 
  loc=df[df['data'].str.contains(total, na=False)]
  t =loc.data.str.split(':', expand=True).pop(1)
  age=t.str.split('ms', expand=True).pop(0)
  frameage['frameage']=age
  frameage.reset_index(inplace=True, drop=True)
  #framesize	
  str_choice = "Frame:" 
  loc=df[df['data'].str.contains(str_choice, na=False)]
  Frame=loc.data.str.split(',', expand=True).pop(1)
  framesize=Frame.str.split('bytes', expand=True).pop(0)
  framesize.reset_index(inplace=True, drop=True)
  frameage['framesize']=framesize
  #cap_time
  str_choice = "CaptureEnd" 
  loc=df[df['data'].str.contains(str_choice, na=False)]
  cap=loc.data.str.split('delta:', expand=True).pop(1)
  Capture=cap.str.split('ms', expand=True).pop(0)
  Capture.reset_index(inplace=True, drop=True)
  frameage['cap_time']=Capture
  #con_time
  str_choice = "ConvertEnd" 
  loc=df[df['data'].str.contains(str_choice, na=False)]
  con=loc.data.str.split('delta:', expand=True).pop(1)
  Convert=con.str.split('ms', expand=True).pop(0)
  Convert.reset_index(inplace=True, drop=True)
  frameage['con_time']=Convert
  #Encode_time
  str_choice = "EncodeEnd" 
  loc=df[df['data'].str.contains(str_choice, na=False)]
  Encode=loc.data.str.split('delta:', expand=True).pop(1)
  EncodeEnd=Encode.str.split('ms', expand=True).pop(0)
  EncodeEnd.reset_index(inplace=True, drop=True)
  frameage['Encode_time']=EncodeEnd
  #transfer_time
  str_choice = "EventRecv" 
  loc=df[df['data'].str.contains(str_choice, na=False)]
  Event=loc.data.str.split('delta:', expand=True).pop(1)
  EventRecv=Event.str.split('ms', expand=True).pop(0)
  EventRecv.reset_index(inplace=True, drop=True)
  frameage['transfer_time']=EventRecv

  #upload_time
  str_choice = "UploadEnd" 
  loc=df[df['data'].str.contains(str_choice, na=False)]
  Upload=loc.data.str.split('delta:', expand=True).pop(1)
  Upload=Upload.str.split('ms', expand=True).pop(0)
  Upload.reset_index(inplace=True, drop=True)
  frameage['Upload_time']=Upload
  #Decode_time
  str_choice = "DecodeEnd" 
  loc=df[df['data'].str.contains(str_choice, na=False)]
  Decode=loc.data.str.split('delta:', expand=True).pop(1)
  DecodeEnd=Decode.str.split('ms', expand=True).pop(0)
  DecodeEnd.reset_index(inplace=True, drop=True)
  frameage['Decode_time']=DecodeEnd

  #complete_time
  str_choice = "EventComplete" 
  loc=df[df['data'].str.contains(str_choice, na=False)]
  Complete=loc.data.str.split('delta:', expand=True).pop(1)
  complete_time=Complete.str.split('ms', expand=True).pop(0)
  complete_time.reset_index(inplace=True, drop=True)
  frameage['complete_time']=complete_time
  #轉成float
  frameage['frameage'] = pd.to_numeric(frameage ['frameage'], errors='coerce')
  frameage['framesize'] = pd.to_numeric(frameage ['framesize'], errors='coerce')
  frameage['cap_time'] = pd.to_numeric(frameage ['cap_time'], errors='coerce')
  frameage['con_time'] = pd.to_numeric(frameage ['con_time'], errors='coerce')
  frameage['Encode_time'] = pd.to_numeric(frameage ['Encode_time'], errors='coerce')
  frameage['transfer_time'] = pd.to_numeric(frameage ['transfer_time'], errors='coerce')
  frameage['Decode_time'] = pd.to_numeric(frameage ['Decode_time'], errors='coerce')
  frameage['Upload_time'] = pd.to_numeric(frameage ['Upload_time'], errors='coerce')
  frameage['complete_time'] = pd.to_numeric(frameage ['complete_time'], errors='coerce')
  frameage=frameage.fillna(0) 

  alldata=net.join(frameage, how='left')
  qos = {'Avg.pingTime':alldata['ping_new'].mean(), "st.Dev.pingTime": alldata['ping_new'].std(),
          'Avg.serverBW':alldata['server_new'].mean(), "St.Dev.serverBW": alldata['server_new'].std(),
          'Avg.clientBW':alldata['client_new'].mean(), "St.Dev.clientBW": alldata['client_new'].std(),
          'Avg.linkBW':alldata['link_new'].mean(), "St.Dev.linkBW": alldata['link_new'].std(),
          'Avg.packetloss':alldata['packetloss_new'].mean(), "St.Dev.packetloss": alldata['packetloss_new'].std(),
          'Avg.FrameAge':alldata['frameage'].mean(), "st.Dev.FrameAge": alldata['frameage'].std(),
          'Avg.FrameSize':alldata['framesize'].mean(), "st.Dev.FrameSize": alldata['framesize'].std(),
          'Avg.CaptureTime':alldata['cap_time'].mean(), "st.Dev.CaptureTime": alldata['cap_time'].std(),
          'Avg.ConvertTime':alldata['con_time'].mean(), "st.Dev.ConvertTime": alldata['con_time'].std(),
          'Avg.EncodeTime':alldata['Encode_time'].mean(), "st.Dev.EncodeTime": alldata['Encode_time'].std(),
          'Avg.TransferTime':alldata['transfer_time'].mean(), "st.Dev.TransferTime": alldata['transfer_time'].std(),
          'Avg.DecodeTime':alldata['Decode_time'].mean(), "st.Dev.DecodeTime": alldata['Decode_time'].std(),
          'Avg.UploadTime':alldata['Upload_time'].mean(), "st.Dev.UploadTime": alldata['Upload_time'].std(),
          'Avg.CompleteTime':alldata['complete_time'].mean(), "st.Dev.CompleteTime": alldata['complete_time'].std()
          }
  sts_cor_qos = pd.DataFrame(qos, index = [0])
  sts_cor_qos=sts_cor_qos.round(3)
  return sts_cor_qos
```

## spilt each user  time stamp

- import .excel file

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#後側問卷
back = pd.read_excel('/content/back_mos2.xlsx')
#修改欄位名稱
back.rename(columns={'時間戳記': 'endtime','座位編號': 'seat','姓名': 'name'},inplace=True) 
#前測問卷
front = pd.read_excel('/content/front_mos2.xlsx')
front.rename(columns={'時間戳記': 'starttime','受測時的座位編號 (張貼於座位左側)': 'seat','姓名': 'name'},inplace=True) #修改欄位名稱

```

- merge

```python
#merge
f=front[['starttime','seat','name']]
b=back[['endtime','name']]
time=pd.merge(f, b, on='name')
time
```

- convert starttime to time format, and conversion time to vm time

```python
from datetime import datetime, timedelta

time["starttime_new"] = time['starttime'] - timedelta(hours=8)
time["endtime_new"] = time['endtime'] - timedelta(hours=8)

time.to_csv('time.csv')
```

## user timestamp map to log file

```python
#匯入受試者體驗時間對照表
datatime = pd.read_csv('/content/time2.csv')
datatime["starttime_new"] = pd.to_datetime(datatime["starttime_new"])
datatime["endtime_new"] = pd.to_datetime(datatime["endtime_new"])

#抓第一個位置
datatime = datatime[datatime['seat']==1.0]
datatime=datatime.drop(labels=['level_0','index','Unnamed: 0'],axis='columns') #刪除欄位

#列出vm1裏所有檔案
b = os.listdir(r"/content/qoestorage/vm1/log")
#排列檔案順序
b.sort(key = lambda x:x.split('.')[0][16:50].replace('_','').replace('(','').replace(')','').replace('-',''))

#file name convert to datetime format
import time,datetime
r = datetime.datetime.strptime(b[5][32:37]+'-2022-'+b[5][41:49],"%m-%d-%Y-%H-%M-%S")

#抓出所有受試者的所有log檔
# 2-6位受試者沒有資料
w=0
player_log = pd.DataFrame()
for i in b[3:]:
  for t in range(1,21):
    r = datetime.datetime.strptime(i[32:37]+'-2022-'+i[41:49],"%m-%d-%Y-%H-%M-%S")
    if datatime['starttime_new'][t]<r and datatime['endtime_new'][t]>r:
      temp_dict = {"user":datatime['name'][t],"file":i}
      df_temp = pd.DataFrame(temp_dict,index = [0])
      player_log = player_log.append(df_temp)
  
      w+=1

player_log
```

## log file

- 排資料的順序

```python
import os
b = os.listdir(r"/content/qoestorage/vm4/log")
b.sort(key = lambda x:x.split('.')[0][16:50].replace('_','').replace('(','').replace(')','').replace('-',''))
```

- 抓數值

```python
#算每一個log參數的AVG,STD(修正版)
#num test
#使用 get_nums() 函數
# 載入現成套件

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import nums_from_string as nfs

def network(file):
  Path = "/content/qoestorage/vm4/log/"
  df = pd.read_csv(Path + file ,delimiter="\t",names=['data']) #將資料變成dataframe
  # 先切資料
  df[['ping', 'server', 'client','link','packetloss']] = df.data.str.split(',', expand=True)

  #把data丟掉
  data2 = df.drop(labels=['data'],axis='columns')
  #顯示出NETWORK開頭的資料
  net_sta2=data2[data2.ping.str.startswith('NETWORK')] 

  net_sta2.reset_index(inplace=True, drop=True)

  ping_list = []
  server_list = []
  client_list = []
  link_list = []
  packetloss_list = []

  for i in range(len(net_sta2)):
    ping = net_sta2.iloc[i, 0]
    ping_new = nfs.get_nums(ping)
    #多個list轉在一起
    ping_num = [str(integer) for integer in ping_new]
    ping_nums = float("".join(ping_num))
    ping_list.append(ping_nums)

    server = net_sta2.iloc[i, 1]
    server_new = nfs.get_nums(server)
    server_num = [str(integer) for integer in server_new]
    server_nums = float("".join(server_num))
    server_list.append(server_nums)

    client = net_sta2.iloc[i, 2]
    client_new = nfs.get_nums(client)
    client_num = [str(integer) for integer in client_new]
    client_nums = float("".join(client_num))
    client_list.append(client_nums)

    link = net_sta2.iloc[i, 3]
    link_new = nfs.get_nums(link)
    link_num = [str(integer) for integer in link_new]
    link_nums = float("".join(link_num))
    link_list.append(link_nums)

    packetloss = net_sta2.iloc[i, 4]
    packetloss_new = nfs.get_nums(packetloss)
    packetloss_num = [str(integer) for integer in packetloss_new]
    packetloss_nums = float("".join(packetloss_num))
    packetloss_list.append(packetloss_nums)

  net_sta2.insert(5, 'ping_new', ping_list)
  net_sta2.insert(6, 'server_new', server_list)
  net_sta2.insert(7, 'client_new', client_list)
  net_sta2.insert(8, 'link_new', link_list)
  net_sta2.insert(9, 'packetloss_new', packetloss_list)
  net= net_sta2.drop(labels=['ping','server','client','link','packetloss'],axis='columns')

  #all frame data
  #抓frameage的資料
  frameage = pd.DataFrame(columns = ["frameage", "framesize", "cap_time", "con_time","Encode_time", "transfer_time", "Decode_time","Upload_time", "complete_time"])
  total = "total" 
  loc=df[df['data'].str.contains(total, na=False)]
  t =loc.data.str.split(':', expand=True).pop(1)
  age=t.str.split('ms', expand=True).pop(0)
  frameage['frameage']=age
  frameage.reset_index(inplace=True, drop=True)

  #framesize	
  str_choice = "Frame:" 
  loc=df[df['data'].str.contains(str_choice, na=False)]
  Frame=loc.data.str.split(',', expand=True).pop(1)
  framesize=Frame.str.split('bytes', expand=True).pop(0)
  framesize.reset_index(inplace=True, drop=True)
  frameage['framesize']=framesize

  #cap_time
  str_choice = "CaptureEnd" 
  loc=df[df['data'].str.contains(str_choice, na=False)]
  cap=loc.data.str.split('delta:', expand=True).pop(1)
  Capture=cap.str.split('ms', expand=True).pop(0)
  Capture.reset_index(inplace=True, drop=True)
  frameage['cap_time']=Capture

  #con_time
  str_choice = "ConvertEnd" 
  loc=df[df['data'].str.contains(str_choice, na=False)]
  con=loc.data.str.split('delta:', expand=True).pop(1)
  Convert=con.str.split('ms', expand=True).pop(0)
  Convert.reset_index(inplace=True, drop=True)
  frameage['con_time']=Convert

  #Encode_time
  str_choice = "EncodeEnd" 
  loc=df[df['data'].str.contains(str_choice, na=False)]
  Encode=loc.data.str.split('delta:', expand=True).pop(1)
  EncodeEnd=Encode.str.split('ms', expand=True).pop(0)
  EncodeEnd.reset_index(inplace=True, drop=True)
  frameage['Encode_time']=EncodeEnd

  #transfer_time
  str_choice = "EventRecv" 
  loc=df[df['data'].str.contains(str_choice, na=False)]
  Event=loc.data.str.split('delta:', expand=True).pop(1)
  EventRecv=Event.str.split('ms', expand=True).pop(0)
  EventRecv.reset_index(inplace=True, drop=True)
  frameage['transfer_time']=EventRecv

  #upload_time
  str_choice = "UploadEnd" 
  loc=df[df['data'].str.contains(str_choice, na=False)]
  Upload=loc.data.str.split('delta:', expand=True).pop(1)
  Upload=Upload.str.split('ms', expand=True).pop(0)
  Upload.reset_index(inplace=True, drop=True)
  frameage['Upload_time']=Upload

  #Decode_time
  str_choice = "DecodeEnd" 
  loc=df[df['data'].str.contains(str_choice, na=False)]
  Decode=loc.data.str.split('delta:', expand=True).pop(1)
  DecodeEnd=Decode.str.split('ms', expand=True).pop(0)
  DecodeEnd.reset_index(inplace=True, drop=True)
  frameage['Decode_time']=DecodeEnd

  #complete_time
  str_choice = "EventComplete" 
  loc=df[df['data'].str.contains(str_choice, na=False)]
  Complete=loc.data.str.split('delta:', expand=True).pop(1)
  complete_time=Complete.str.split('ms', expand=True).pop(0)
  complete_time.reset_index(inplace=True, drop=True)
  frameage['complete_time']=complete_time
  
  #轉成float
  frameage['frameage'] = pd.to_numeric(frameage ['frameage'], errors='coerce')
  frameage['framesize'] = pd.to_numeric(frameage ['framesize'], errors='coerce')
  frameage['cap_time'] = pd.to_numeric(frameage ['cap_time'], errors='coerce')
  frameage['con_time'] = pd.to_numeric(frameage ['con_time'], errors='coerce')
  frameage['Encode_time'] = pd.to_numeric(frameage ['Encode_time'], errors='coerce')
  frameage['transfer_time'] = pd.to_numeric(frameage ['transfer_time'], errors='coerce')
  frameage['Decode_time'] = pd.to_numeric(frameage ['Decode_time'], errors='coerce')
  frameage['Upload_time'] = pd.to_numeric(frameage ['Upload_time'], errors='coerce')
  frameage['complete_time'] = pd.to_numeric(frameage ['complete_time'], errors='coerce')
  frameage=frameage.fillna(0) 

  alldata=net.join(frameage, how='left')
  qos = {'Avg.pingTime':alldata['ping_new'].mean(), "st.Dev.pingTime": alldata['ping_new'].std(),
          'Avg.serverBW':alldata['server_new'].mean(), "St.Dev.serverBW": alldata['server_new'].std(),
          'Avg.clientBW':alldata['client_new'].mean(), "St.Dev.clientBW": alldata['client_new'].std(),
          'Avg.linkBW':alldata['link_new'].mean(), "St.Dev.linkBW": alldata['link_new'].std(),
          'Avg.packetloss':alldata['packetloss_new'].mean(), "St.Dev.packetloss": alldata['packetloss_new'].std(),
          'Avg.FrameAge':alldata['frameage'].mean(), "st.Dev.FrameAge": alldata['frameage'].std(),
          'Avg.FrameSize':alldata['framesize'].mean(), "st.Dev.FrameSize": alldata['framesize'].std(),
          'Avg.CaptureTime':alldata['cap_time'].mean(), "st.Dev.CaptureTime": alldata['cap_time'].std(),
          'Avg.ConvertTime':alldata['con_time'].mean(), "st.Dev.ConvertTime": alldata['con_time'].std(),
          'Avg.EncodeTime':alldata['Encode_time'].mean(), "st.Dev.EncodeTime": alldata['Encode_time'].std(),
          'Avg.TransferTime':alldata['transfer_time'].mean(), "st.Dev.TransferTime": alldata['transfer_time'].std(),
          'Avg.DecodeTime':alldata['Decode_time'].mean(), "st.Dev.DecodeTime": alldata['Decode_time'].std(),
          'Avg.UploadTime':alldata['Upload_time'].mean(), "st.Dev.UploadTime": alldata['Upload_time'].std(),
          'Avg.CompleteTime':alldata['complete_time'].mean(), "st.Dev.CompleteTime": alldata['complete_time'].std()
          }
  sts_cor_qos = pd.DataFrame(qos, index = [0])
  sts_cor_qos=sts_cor_qos.round(3)
  return sts_cor_qos
```

- read all log.txt

```python
#txt
import os
x = pd.DataFrame()
b = os.listdir(r"/content/qoestorage/vm4/log")
b.sort(key = lambda x:x.split('.')[0][16:50].replace('_','').replace('(','').replace(')','').replace('-',''))

for i in b:
  try:
    df = network(f"{i}")
    df["file"] = i
    x = x.append(df)
  except:
    print(f"error with the file-{i}")
    pass

x
```

- 加上檔名的欄位

```python
x['file'] = x['file'].str.split('_StreamVideoTrace').str[0]
x=x.reset_index(drop=True)
x
```

- use file name to merge user and log file

```python
seat1=pd.merge(player_log, x, on='file')
seat1.to_csv('seat4.1.csv')
seat1
```

- merge log file and user

```python
# x =Each log AVG and Std
all=pd.merge(player_log, x, on='file')
all.to_csv('all.csv')
```

## heatmap

[https://blog.csdn.net/ztf312/article/details/102474190](https://blog.csdn.net/ztf312/article/details/102474190)

```python
a=relation.corr()
b=a.iloc[0:28,28:32]

```


```python
#相關性分析
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set(context="notebook",style="whitegrid",palette="dark")
plt.subplots(figsize=(15,10)) #設置長寬尺吋大小
c1 = sns.heatmap(b, annot = True, vmax = 1)
# c1 = sns.heatmap(b, annot = True, vmax = 1, cmap="Blues")
plt.savefig("rawrelation_0412.png", dpi=800)
```



## 實驗檔案位置

在本機端跑CNN模型，訓練好的模型放在實驗資料夾裡，檔名0507model.h5
