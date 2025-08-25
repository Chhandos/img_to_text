import tensorflow as tf
import numpy as np
import os
from pickle import dump,load
# import matplotlib.pyplot as plt
from tensorflow.keras.applications.xception import Xception,preprocess_input
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical,get_file
from tensorflow.keras.layers import add
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Input,Dense,LSTM,Embedding,Dropout
from pickle import dump

# from tqdm import tqdm_notebook as tqdm
# tqdm().pandas()

def load_doc(filename):
    file=open(filename,"r")
    text=file.read()
    file.close()
    return text


def all_img_captions(filename):
    file=load_doc(filename)
    rows=file.split("\n")
    rows=[i.split("\t") for i in rows]
    return(rows)

rows=all_img_captions("Flickr8k.lemma.token.txt") 



rows = [i for i in rows if len(i) == 2]  # ✅



Img_X=[]
Cap_X=[]
for i in rows:
   image=i[0]
   path=image.split("#")[0]
   if(path.endswith("1")):
       continue
   Img_X.append(path)
   Cap_X.append(i[1])



import string

def clean_text(text):
    # exclude = string.punctuation.replace("'", "")  # keep only apostrophe
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


#DAta Cleaninh

#cnvert everything to lower case
Cap_X=["startseq " + clean_text(i.lower()[:-2]) + " endseq" for i in Cap_X]
# print(Cap_X)






text=" ".join(Cap_X)
# print(text)


#remove punctuators


# cleaned_text=clean_text(text);

from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer=Tokenizer()

tokenizer.fit_on_texts([text])

dic=tokenizer.word_index
# print(dic[7043])


with open("word_index.pkl","wb") as f:
    dump(dic,f)


print(len(list(dic)))

# Now convert each wrd to indices in sentences by sentence


Cap_X_indexed=[]
# longest_sentence_len=max(Cap_X_indexed,key=len)



Cap_X_indexed = [tokenizer.texts_to_sequences([s])[0] for s in Cap_X]

#Now find maxlen for padding

max_len=max(len(i) for i in Cap_X_indexed)
print(max_len) 
# Cap_X_padded=pad_sequences(Cap_X_indexed,maxlen=max_len,padding="pre")

# print(Cap_X_padded)

# 
# with open("captions_padded.pkl", "wb") as f:
#     dump(Cap_X_padded, f)





#Text Processing Done






# Img_X=[i[:-2] for i in Img_X]
# print("Img_X",Img_X)



unique_Img_X = list(dict.fromkeys(Img_X))
# print(unique_Img_X)
# print(len(unique_Img_X))


img_x_cap={}
count=0
for i in range (len(unique_Img_X)):
    l=[]
    for j in range(5):
        for k in range (1,len(Cap_X_indexed[count])):
            temp=Cap_X_indexed[count][:k+1]
            temp=pad_sequences([temp],maxlen=max_len,padding="pre")
            l.append(temp)
        
        # l.append(Cap_X_padded[count])
        count+=1
    img_x_cap[unique_Img_X[i]]=l

# print(img_x_cap)

with open("img_x_capindex.pkl", "wb") as f:
    dump(img_x_cap, f)



# print(img_x_cap["1305564994_00513f9a5b.jpg"])    




#load Xception model(Transfer leaerning)

model = Xception(weights='imagenet', include_top=False,pooling="avg")

from tensorflow.keras.preprocessing.image import load_img, img_to_array

#get images



path_array=[]

# folder = "Flicker8k_Dataset"

# for fname in os.listdir(folder):
#     if fname.endswith(".1") or ".jpg." in fname:
#         print("Deleting:", fname)
#         os.remove(os.path.join(folder, fname))



folder = "Flicker8k_Dataset"
for i in unique_Img_X:
    imgpath = os.path.join(folder, i)
    # print(imgpath)
    path_array.append(imgpath)
    
    
    # Images_array.append(image)


#     Images_array.append(image)

# # print(Images_array[0])


img_array=[]   
for i in path_array:
    # print(i)
    image = load_img(i, target_size=(299, 299))  # shape: (299, 299, 3)
    image = img_to_array(image)
    image = preprocess_input(image)  #  Scaled to [-1, 1]
    img_array.append(image)

img_array=np.array(img_array) 
# print(img_array.shape)


img_feature_matrices=model.predict(img_array)#extract feature mtrices of images
# print(img_feature_matrices)




#create dictionary

image_x_feature={}

for i in range (len(path_array)):
    name = os.path.basename(path_array[i])
    image_x_feature[name]=img_feature_matrices[i]
    



#save the featureectors in file for reuseability
with open("xception_features.pkl", "wb") as f:
    dump(image_x_feature, f)















