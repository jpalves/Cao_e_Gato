import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from keras.models import load_model
model = load_model('model1_catsVSdogs_100_epoch.h5')
#dictionary to label all traffic signs class.
#initialise GUI
top=tk.Tk()
top.geometry('800x600')
top.title('Classificação Cão ou Gato')
top.configure(background='#CDCDCD')
label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)
def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((224,224))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    image = image/255
    pred = model.predict([image])[0]
    #sign = classes[pred]
    #print(sign)
    inds = pred.argsort()[::-1][:5]
    #print(inds)
    cls_list = ['gato', 'cão']
    for i in inds:
        print('    {:.3f}  {}'.format(pred[i], cls_list[i]))
    label.configure(foreground='#011638', text=cls_list[np.argmax(pred)]) 
def show_classify_button(file_path):
    classify_b=Button(top,text="Classifica",
   command=lambda: classify(file_path),
   padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',
font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),
    (top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass
upload=Button(top,text="Upload uma imagem",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Cats VS Dogs",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()
