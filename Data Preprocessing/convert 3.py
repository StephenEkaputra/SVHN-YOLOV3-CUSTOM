import os
from os import walk, getcwd
from PIL import Image

classes = ["1","2","3","4","5","6","7","8","9","10"]

def convert(size, box):
    dw = size[0]
    dh = size[1]
    x = box[0] + (box[1]/2.0)
    y = box[2] + (box[3]/2.0)
    w = box[1]
    h = box[3]
    x = x/dw
    w = w/dw
    y = y/dh
    h = h/dh
    return (x,y,w,h)
    
    
"""-------------------------------------------------------------------""" 

""" Configure Paths"""   
mypath = "Labels/001/"
outpath = "Input/"

cls = "1"
if cls not in classes:
    exit(0)
cls_id = classes.index(cls)

wd = getcwd()
list_file = open('%s/%s_list.txt'%(wd, cls), 'w')

""" Get input text file list """
txt_name_list = []
for (dirpath, dirnames, filenames) in walk(mypath):
    txt_name_list.extend(filenames)
    break
print(txt_name_list)
txt_name_list = sorted(txt_name_list,key=lambda x: int(os.path.splitext(x)[0]))

""" Process """
for txt_name in txt_name_list:
    
    """ Open input text files """
    txt_path = mypath + txt_name
    print("Input:" + txt_path)
    txt_file = open(txt_path, "r")
    lines = txt_file.read().split('\n')   #for ubuntu, use "\r\n" instead of "\n"
    
    """ Open output text files """
    txt_outpath = outpath + txt_name
    print("Output:" + txt_outpath)
    txt_outfile = open(txt_outpath, "w")
    
    
    """ Convert the data to YOLO format """
    ct = 0
    for line in lines:
        if(len(line) >= 2):
            ct = ct + 1
            print(line + "\n")
            elems = line.split(' ')
            print(elems)
            xmin = elems[0]
            xmax = elems[2]
            ymin = elems[1]
            ymax = elems[3]
            cls = elems[4]
            
            img_path = str('Input/%s.png'%(os.path.splitext(txt_name)[0]))
            
            im=Image.open(img_path)
            w= int(im.size[0])
            h= int(im.size[1])
            
            print('W H = ', w, h)
            b = (float(xmin), float(xmax), float(ymin), float(ymax))
            print('B = ', b)
            bb = convert((w,h), b)
            print(bb)
            print("CLS = ", cls)
            txt_outfile.write(str(cls) + " " + " ".join([str(a) for a in bb]) + '\n')

    """ Save those images with bb into list"""
    if(ct != 0):
        list_file.write('Input/%s.png\n'%(os.path.splitext(txt_name)[0]))
                
list_file.close()       

