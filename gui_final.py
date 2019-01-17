# -*- coding: utf-8 -*-
"""

"""
from PIL import Image, ImageTk
from resnet50_test_model import compute_and_return
import tkinter as tk
import tkinter.filedialog
import os
import numpy as np

pathVariable = 'person.jpg'
g_path = 'person.jpg'
window = tk.Tk()
window.title("dog-retrival")
window.geometry('1600x600')
dog_breed = ['Chihuahua','Chow','Doberman','French_bulldog','German_shepherd','Golden_retriever','Labrador_retriever','Malamute','Papillon','Pekinese','Pomeranian','Pug','Samoyed','Shih-Tzu','Siberian_husky','Standard_poodle','Standard_schnauzer','Tibetan_mastiff']
width = 180
height = 240

def SelectPath():
	
	
    global g_path
    g_path = tk.filedialog.askopenfilename()
    path.set(g_path)

    # os.system('cd /home/cv_wbchen/YOLO_v3_tutorial_from_scratch && python /home/cv_wbchen/YOLO_v3_tutorial_from_scratch/detect.py  --images' +' '+ g_path)
    os.system('python detect.py  --images' +' '+ g_path)

		
    global g_crop_path
    g_crop_path = './new_pic/det_crop' + g_path.split('/')[-1]
    global g_img1
    g_img1 = Image.open(g_crop_path)
    g_img1 = g_img1.resize((width,height), Image.ANTIALIAS)
    global g_photo1
    g_photo1 = ImageTk.PhotoImage(g_img1)
    g_imgLabel1.configure(image=g_photo1)  

def Compute():

		
    global result, dog_type
    result, dog_type = compute_and_return(g_crop_path)
    global lab1
    lab1 = 'Pred: ' + dog_breed[np.argmax(dog_type[0])]
    g_lab1.configure(text = lab1)
   
    global g_img2
    g_img2 = Image.open(result[0])
    g_img2 = g_img2.resize((width,height), Image.ANTIALIAS)
    global g_photo2
    g_photo2 = ImageTk.PhotoImage(g_img2)
    g_imgLabel2.configure(image=g_photo2)
    global lab2
    lab2 = (result[0].split('/')[-1]).split('.')[0]
    g_lab2.configure(text = lab2)

    global g_img3
    g_img3 = Image.open(result[1])
    g_img3 = g_img3.resize((width,height), Image.ANTIALIAS)
    global g_photo3
    g_photo3 = ImageTk.PhotoImage(g_img3)
    g_imgLabel3.configure(image=g_photo3)
    global lab3
    lab3 = (result[1].split('/')[-1]).split('.')[0]
    g_lab3.configure(text = lab3)

    global g_img4
    g_img4 = Image.open(result[2])
    g_img4 = g_img4.resize((width,height), Image.ANTIALIAS)
    global g_photo4
    g_photo4 = ImageTk.PhotoImage(g_img4)
    g_imgLabel4.configure(image=g_photo4)
    global lab4
    lab4 = (result[2].split('/')[-1]).split('.')[0]
    g_lab4.configure(text = lab4)

    global g_img5
    g_img5 = Image.open(result[3])
    g_img5 = g_img5.resize((width,height), Image.ANTIALIAS)
    global g_photo5
    g_photo5 = ImageTk.PhotoImage(g_img5)
    g_imgLabel5.configure(image=g_photo5)
    global lab5
    lab5 = (result[3].split('/')[-1]).split('.')[0]
    g_lab5.configure(text = lab5)

    global g_img6
    g_img6 = Image.open(result[4])
    g_img6 = g_img6.resize((width,height), Image.ANTIALIAS)
    global g_photo6
    g_photo6 = ImageTk.PhotoImage(g_img6)
    g_imgLabel6.configure(image=g_photo6)
    global lab6
    lab6 = (result[4].split('/')[-1]).split('.')[0]
    g_lab6.configure(text = lab6)

    global g_img7
    g_img7 = Image.open(result[5])
    g_img7 = g_img7.resize((width,height), Image.ANTIALIAS)
    global g_photo7
    g_photo7 = ImageTk.PhotoImage(g_img7)
    g_imgLabel7.configure(image=g_photo7)
    global lab7
    lab7 = (result[5].split('/')[-1]).split('.')[0]
    g_lab7.configure(text = lab7)

    global g_img8
    g_img8 = Image.open(result[6])
    g_img8 = g_img8.resize((width,height), Image.ANTIALIAS)
    global g_photo8
    g_photo8 = ImageTk.PhotoImage(g_img8)
    g_imgLabel8.configure(image=g_photo8)
    global lab8
    lab8 = (result[6].split('/')[-1]).split('.')[0]
    g_lab8.configure(text = lab8)

    global g_img9
    g_img9 = Image.open(result[7])
    g_img9 = g_img9.resize((width,height), Image.ANTIALIAS)
    global g_photo9
    g_photo9 = ImageTk.PhotoImage(g_img9)
    g_imgLabel9.configure(image=g_photo9)
    global lab9
    lab9 = (result[7].split('/')[-1]).split('.')[0]
    g_lab9.configure(text = lab9)

    global g_img10
    g_img10 = Image.open(result[8])
    g_img10 = g_img10.resize((width,height), Image.ANTIALIAS)
    global g_photo10
    g_photo10 = ImageTk.PhotoImage(g_img10)
    g_imgLabel10.configure(image=g_photo10)
    global lab10
    lab10 = (result[8].split('/')[-1]).split('.')[0]
    g_lab10.configure(text = lab10)

    global g_img11
    g_img11 = Image.open(result[9])
    g_img11 = g_img11.resize((width,height), Image.ANTIALIAS)
    global g_photo11
    g_photo11 = ImageTk.PhotoImage(g_img11)
    g_imgLabel11.configure(image=g_photo11)
    global lab11
    lab11 = (result[9].split('/')[-1]).split('.')[0]
    g_lab11.configure(text = lab11)

    global g_img12
    g_img12 = Image.open(result[10])
    g_img12 = g_img12.resize((width,height), Image.ANTIALIAS)
    global g_photo12
    g_photo12 = ImageTk.PhotoImage(g_img12)
    g_imgLabel12.configure(image=g_photo12)
    global lab12
    lab12 = (result[10].split('/')[-1]).split('.')[0]
    g_lab12.configure(text = lab12)

    global g_img13
    g_img13 = Image.open(result[11])
    g_img13 = g_img13.resize((width,height), Image.ANTIALIAS)
    global g_photo13
    g_photo13 = ImageTk.PhotoImage(g_img13)
    g_imgLabel13.configure(image=g_photo13)
    global lab13
    lab13 = (result[11].split('/')[-1]).split('.')[0]
    g_lab13.configure(text = lab13)
    
    global g_img14
    g_img14 = Image.open(result[12])
    g_img14 = g_img14.resize((width,height), Image.ANTIALIAS)
    global g_photo14
    g_photo14 = ImageTk.PhotoImage(g_img14)
    g_imgLabel14.configure(image=g_photo14)
    global lab14
    lab14 = (result[12].split('/')[-1]).split('.')[0]
    g_lab14.configure(text = lab14)

    global g_img15
    g_img15 = Image.open(result[13])
    g_img15 = g_img15.resize((width,height), Image.ANTIALIAS)
    global g_photo15
    g_photo15 = ImageTk.PhotoImage(g_img15)
    g_imgLabel15.configure(image=g_photo15)
    global lab15
    lab15 = (result[13].split('/')[-1]).split('.')[0]
    g_lab15.configure(text = lab15)

    global g_img16
    g_img16 = Image.open(result[14])
    g_img16 = g_img16.resize((width,height), Image.ANTIALIAS)
    global g_photo16
    g_photo16 = ImageTk.PhotoImage(g_img16)
    g_imgLabel16.configure(image=g_photo16)
    global lab16
    lab16 = (result[14].split('/')[-1]).split('.')[0]
    g_lab16.configure(text = lab16)


path = tk.StringVar()


l1 = tk.Label(window, text="Target:", width=10, height=2).grid(row=0, column=0)
e = tk.Entry(window, textvariable=path, width=80).grid(row=0, column=1, columnspan=3)
b1 = tk.Button(window, text="Select", command=SelectPath).grid(row=0, column=5)
b2 = tk.Button(window, text="Compute", command=Compute).grid(row=0, column=6)

g_img1 = Image.open('qiep_01.jpg')
g_img1 = g_img1.resize((width,height), Image.ANTIALIAS)


g_photo1 = ImageTk.PhotoImage(g_img1)
g_imgLabel1 = tk.Label(window, image=g_photo1, width=width, height=height)
g_imgLabel1.grid(row=1, column=0)
lab1 = '?'
g_lab1 = tk.Label(window, text = 'Pred:' + lab1)
g_lab1.grid(row=2, column=0)


g_img2 = Image.open('qiep_02.jpg')
g_img2 = g_img2.resize((width,height), Image.ANTIALIAS)
g_photo2 = ImageTk.PhotoImage(g_img2)
g_imgLabel2 = tk.Label(window, image=g_photo2, width=width, height=height)
g_imgLabel2.grid(row=1, column=1)
lab2 = 'image1'
g_lab2 = tk.Label(window, text = lab2)
g_lab2.grid(row=2, column=1)


g_img3 = Image.open('qiep_03.jpg')
g_img3 = g_img3.resize((width,height), Image.ANTIALIAS)
g_photo3 = ImageTk.PhotoImage(g_img3)
g_imgLabel3 = tk.Label(window, image=g_photo3, width=width, height=height)
g_imgLabel3.grid(row=1, column=2)
lab3 = 'image2'
g_lab3 = tk.Label(window, text = lab3)
g_lab3.grid(row=2, column=2)

g_img4 = Image.open('qiep_04.jpg')
g_img4 = g_img4.resize((width,height), Image.ANTIALIAS)
g_photo4 = ImageTk.PhotoImage(g_img4)
g_imgLabel4 = tk.Label(window, image=g_photo4, width=width, height=height)
g_imgLabel4.grid(row=1, column=3)
lab4 = 'image3'
g_lab4 = tk.Label(window, text = lab4)
g_lab4.grid(row=2, column=3)

g_img5 = Image.open('qiep_05.jpg')
g_img5 = g_img5.resize((width,height), Image.ANTIALIAS)
g_photo5 = ImageTk.PhotoImage(g_img5)
g_imgLabel5 = tk.Label(window, image=g_photo5, width=width, height=height)
g_imgLabel5.grid(row=1, column=4)
lab5 = 'image4'
g_lab5 = tk.Label(window, text = lab5)
g_lab5.grid(row=2, column=4)

g_img6 = Image.open('qiep_06.jpg')
g_img6 = g_img6.resize((width,height), Image.ANTIALIAS)
g_photo6 = ImageTk.PhotoImage(g_img6)
g_imgLabel6 = tk.Label(window, image=g_photo6, width=width, height=height)
g_imgLabel6.grid(row=1, column=5)
lab6 = 'image5'
g_lab6 = tk.Label(window, text = lab6)
g_lab6.grid(row=2, column=5)

g_img7 = Image.open('qiep_07.jpg')
g_img7 = g_img7.resize((width,height), Image.ANTIALIAS)
g_photo7 = ImageTk.PhotoImage(g_img7)
g_imgLabel7 = tk.Label(window, image=g_photo7, width=width, height=height)
g_imgLabel7.grid(row=1, column=6)
lab7 = 'image6'
g_lab7 = tk.Label(window, text = lab7)
g_lab7.grid(row=2, column=6)

g_img8 = Image.open('qiep_08.jpg')
g_img8 = g_img8.resize((width,height), Image.ANTIALIAS)
g_photo8 = ImageTk.PhotoImage(g_img8)
g_imgLabel8 = tk.Label(window, image=g_photo8, width=width, height=height)
g_imgLabel8.grid(row=1, column=7)
lab8 = 'image7'
g_lab8 = tk.Label(window, text = lab8)
g_lab8.grid(row=2, column=7)

g_img9 = Image.open('qiep_09.jpg')
g_img9 = g_img9.resize((width,height), Image.ANTIALIAS)
g_photo9 = ImageTk.PhotoImage(g_img9)
g_imgLabel9 = tk.Label(window, image=g_photo9, width=width, height=height)
g_imgLabel9.grid(row=3, column=0)
lab9 = 'image8'
g_lab9 = tk.Label(window, text = lab9)
g_lab9.grid(row=4, column=0)

g_img10 = Image.open('qiep_10.jpg')
g_img10 = g_img10.resize((width,height), Image.ANTIALIAS)
g_photo10 = ImageTk.PhotoImage(g_img10)
g_imgLabel10 = tk.Label(window, image=g_photo10, width=width, height=height)
g_imgLabel10.grid(row=3, column=1)
lab10 = 'image9'
g_lab10 = tk.Label(window, text = lab10)
g_lab10.grid(row=4, column=1)

g_img11 = Image.open('qiep_11.jpg')
g_img11 = g_img11.resize((width,height), Image.ANTIALIAS)
g_photo11 = ImageTk.PhotoImage(g_img11)
g_imgLabel11 = tk.Label(window, image=g_photo11, width=width, height=height)
g_imgLabel11.grid(row=3, column=2)
lab11 = 'image10'
g_lab11 = tk.Label(window, text = lab11)
g_lab11.grid(row=4, column=2)

g_img12 = Image.open('qiep_12.jpg')
g_img12 = g_img12.resize((width,height), Image.ANTIALIAS)
g_photo12 = ImageTk.PhotoImage(g_img12)
g_imgLabel12 = tk.Label(window, image=g_photo12, width=width, height=height)
g_imgLabel12.grid(row=3, column=3)
lab12 = 'image11'
g_lab12 = tk.Label(window, text = lab12)
g_lab12.grid(row=4, column=3)

g_img13 = Image.open('qiep_13.jpg')
g_img13 = g_img13.resize((width,height), Image.ANTIALIAS)
g_photo13 = ImageTk.PhotoImage(g_img13)
g_imgLabel13 = tk.Label(window, image=g_photo13, width=width, height=height)
g_imgLabel13.grid(row=3, column=4)
lab13 = 'image12'
g_lab13 = tk.Label(window, text = lab13)
g_lab13.grid(row=4, column=4)

g_img14 = Image.open('qiep_14.jpg')
g_img14 = g_img14.resize((width,height), Image.ANTIALIAS)
g_photo14 = ImageTk.PhotoImage(g_img14)
g_imgLabel14 = tk.Label(window, image=g_photo14, width=width, height=height)
g_imgLabel14.grid(row=3, column=5)
lab14 = 'image13'
g_lab14 = tk.Label(window, text = lab14)
g_lab14.grid(row=4, column=5)

g_img15 = Image.open('qiep_15.jpg')
g_img15 = g_img15.resize((width,height), Image.ANTIALIAS)
g_photo15 = ImageTk.PhotoImage(g_img15)
g_imgLabel15 = tk.Label(window, image=g_photo15, width=width, height=height)
g_imgLabel15.grid(row=3, column=6)
lab15 = 'image14'
g_lab15 = tk.Label(window, text = lab15)
g_lab15.grid(row=4, column=6)

g_img16 = Image.open('qiep_16.jpg')
g_img16 = g_img16.resize((width,height), Image.ANTIALIAS)
g_photo16 = ImageTk.PhotoImage(g_img16)
g_imgLabel16 = tk.Label(window, image=g_photo16, width=width, height=height)
g_imgLabel16.grid(row=3, column=7)
lab16 = 'image15'
g_lab16 = tk.Label(window, text = lab16)
g_lab16.grid(row=4, column=7)

window.mainloop()
