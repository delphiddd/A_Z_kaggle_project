#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


df = pd.read_csv('archive/A_Z.csv')


# In[4]:


df


# In[5]:


df.keys()


# In[6]:


import numpy as np


# In[7]:


Y = df.iloc[: ,0]


# In[8]:


Y


# In[9]:


Y.shape


# In[10]:


X = df.iloc[: , 1:]


# In[11]:


X


# In[12]:


sample = len(Y)


# In[13]:


X = np.array(X).reshape(sample , -1)


# In[14]:


X.shape


# In[15]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA

pca = PCA(n_components=100)
X_Reduce = pca.fit_transform(X)


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X_Reduce, Y, test_size=0.4, random_state=42)


# In[17]:


model = SVC()


# In[18]:


model.fit(X_train, y_train)


# In[19]:


y_pred = model.predict(X_test)


# In[26]:


import joblib

joblib.dump(model, 'A_Z_model.pkl')


# In[20]:


from sklearn.metrics import accuracy_score, classification_report


# In[21]:


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[110]:


from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

def trans_img(imgfile):
    # เปิดภาพและแปลงเป็นขาวดำ
    img = Image.open(imgfile).convert('L')
    
    # ปรับขนาดภาพเป็น 28x28 พิกเซล
    img = img.resize((28, 28), Image.LANCZOS)
    
    # กลับสีภาพ (invert)
    img = ImageOps.invert(img)
    
    # แปลงภาพเป็น array ของ numpy
    pixel = np.array(img)
    
    # reshape array เพื่อให้มีขนาด (1, 784)
    my_pic = pixel.reshape(1, -1)

    return my_pic

def predict_digit(imgfile):
   for n in imgfile:
        # แปลงภาพ
        my_pic = trans_img(n)
        
        # ลดมิติข้อมูลด้วย PCA ที่ฝึกมาแล้ว
        my_pic_reduced = pca.transform(my_pic)
        
        # ทำนายผลลัพธ์
        predicted = model.predict(my_pic_reduced)
        
        pred_char = chr(predicted[0] + ord('A'))
        
        # แสดงภาพและผลลัพธ์
        plt.figure(figsize=(2, 2))
        plt.imshow(my_pic.reshape(28, 28), cmap='gray')
        plt.title('Predicted: {} => {}'.format(pred_char , predicted))
        plt.xticks([])
        plt.yticks([])
        plt.show()



# In[109]:


imgfile = ['/Users/delphi/Desktop/works/python/ML/38f1426f599056ac59dca1d852246b5d.jpg' , '/Users/delphi/Desktop/works/python/ML/L_Letter_LZNQBD.jpg']
predict_digit(imgfile)


# In[83]:


label_A_Z = {}
for i in range(26):
    label_A_Z[i] = chr(65+i)

print(label_A_Z)


# In[88]:


# แสดงภาพตัวอักษร A-Z พร้อมตัวเลขที่สอดคล้องกัน
plt.figure(figsize=(20, 10))
for i in range(26):
    plt.subplot(3, 9, i + 1)
    # เลือกภาพตัวอย่างของตัวอักษรแต่ละตัว
    idx = np.where(Y == i)[0][0]
    img = X[idx].reshape(28, 28)
    plt.imshow(img, cmap='gray')
    plt.title(i)
    plt.axis('off')

plt.tight_layout()
plt.show()


# In[ ]:




