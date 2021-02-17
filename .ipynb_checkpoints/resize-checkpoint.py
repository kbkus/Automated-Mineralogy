FRAME_HEIGHT = 800
FRAME_WIDTH = 500

import pandas as pd
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import sys

print(sys.version)



class NewFrame(tk.Frame):
    def __init__(self, row, col, img, master=None):
        self.parent = master
        self.row = row
        self.col = col
        self.img = img
        self.pts = []
        tk.Frame.__init__(self, self.parent, bg='green', width=FRAME_WIDTH, height=FRAME_HEIGHT)
        self.grid(row=row, column=col)
        self.canvas = self.__create_canvas()
        self.canvas.bind('<Button-1>', lambda event: self.getxy(self.canvas, event))
    
    def __create_canvas(self):
        width, height = self.img.size
        self.scaler = height/FRAME_HEIGHT
        canvas = tk.Canvas(self.master, bg='yellow', width=FRAME_WIDTH, height=FRAME_HEIGHT)
        canvas.grid(row=self.row, column=self.col, sticky='nesw')
        canvas.image = ImageTk.PhotoImage(self.img.resize((int(np.ceil(width/self.scaler)), FRAME_HEIGHT), Image.ANTIALIAS))
        canvas.create_image(self.row, self.col, image=canvas.image, anchor='nw')
        return canvas

    def getxy(self, canvas, event):
        self.canvas = canvas
        self.pts.append([int(event.x*self.scaler), int(event.y*self.scaler)])
        self.canvas.create_oval(event.x-4, event.y-4, event.x+4, event.y+4, fill='yellow')
        print("pxl location = ({0},{1})".format(int(event.x*self.scaler), int(event.y*self.scaler)))

    def get_pts(self):
        return self.pts

class ExportButton(tk.Frame):
    def __init__(self, frame1, frame2, frame3, master=None):
        self.parent = master
        self.frame1 = frame1
        self.frame2 = frame2
        self.frame3 = frame3

        self.button = tk.Button(master, command=self.buttonClick, text='Export Points')
        self.button.place(x=1550, y=400)

    def buttonClick(self):
        ''' handle button click event to export all of the stored
        click coordinates'''
        print('export button clicked')
        frame1_pts = self.frame1.get_pts()
        frame2_pts = self.frame2.get_pts()
        frame3_pts = self.frame3.get_pts()

        data = {'ppl':frame1_pts,'xpl':frame2_pts, 'labels':frame3_pts}

        df = pd.DataFrame(data)
        df.to_pickle('resize_pts.pkl')




class MainWindow(tk.Frame):
    def __init__(self, img1, img2,img3, master=None):
        self.parent = master
        self.img1 = img1
        self.img2 = img2
        self.img3 = img3

        tk.Frame.__init__(self, self.parent, bg='#ffffff')

        # add quit button
        self.quit_button = tk.Button(master, text='Close Window', command=self.quit)
        self.quit_button.place(x=1550, y=600)

        self.__create_layout()


    def __create_layout(self):
        self.parent.grid()
        self.Frame1 = NewFrame(0, 0, self.img1, self.parent)
        self.Frame2 = NewFrame(0,1, self.img2, self.parent)
        self.Frame3 = NewFrame(0,2, self.img3, self.parent)

        self.Button = ExportButton(self.Frame1, self.Frame2, self.Frame3)

    def quit(self):
        print('quit button pressed')
        self.parent.destroy()
        
    

def main(img1, img2, img3):

    root = tk.Tk()
    
    root.title("Image Ref Window")
    root.geometry("{0}x{1}".format(FRAME_HEIGHT*3, FRAME_HEIGHT))
    # assign cursor appearance
    root.config(cursor="draft_small")

    mw = MainWindow(img1, img2, img3, master=root)
  
    root.mainloop()

    list1 = mw.Frame1.get_pts()
    list2 = mw.Frame2.get_pts()
    list3 = mw.Frame3.get_pts()

    return list1, list2, list3

if __name__ == '__main__':
    ppl = Image.open('/Users/kacikus/Dropbox/AutomatedMineralogy_Project/Automated-Mineralogy/Images/EDF-17-1-PPL.jpg')
    xpl = Image.open('/Users/kacikus/Dropbox/AutomatedMineralogy_Project/Automated-Mineralogy/Images/EDF-17-1-CPL.jpg')
    labels = Image.open('/Users/kacikus/Dropbox/AutomatedMineralogy_Project/Automated-Mineralogy/Images/EDF17-1.png')
    main(ppl, xpl, labels)
