import tkinter as tk
import tkinter.messagebox
from ModelClass import ModelClass
from ScheduleClass import ScheduleClass
from ImageProcess import ImageProcess
from PIL import Image, ImageTk


class GUI(tkinter.Tk):
    """
    基本的GUI界面，能够完成模型训练和图像的检测
    """""
    def __init__(self):
        super().__init__()
        self.filename = None
        self.title("手写数字识别器")
        self.geometry("360x400")
        self.CNN = ModelClass()
        self.Schedule = ScheduleClass()
        self.Img = ImageProcess()
        self.layout()

    def layout(self):
        # 样例验证码预测结果文本框
        self.result_display = tk.Entry(self)
        self.result_display.place(x=100, y=140)
        # 教务处验证码预测结果文本框
        self.verifycode_display = tk.Entry(self)
        self.verifycode_display.place(x=100, y=240)
        # 按钮布局
        self.start_train_button = tk.Button(self, text="开始训练", command=self.do_train)
        self.start_train_button.place(x=140, y=25)
        self.start_predict_button = tk.Button(self, text="开始识别", command=self.do_predict)
        self.start_predict_button.place(x=140, y=70)
        self.pilImage = Image.open("verifycode.jpg")
        self.tkImage = ImageTk.PhotoImage(image=self.pilImage)
        self.verifycode = tk.Label(self, image=self.tkImage)
        self.verifycode.place(x=137, y=100)
        self.get_distance_verifycode = tk.Button(self, text="获取教务处验证码", command=self.place_verifycode)
        self.get_distance_verifycode.place(x=70, y=165)
        self.predict_distance_verifycode = tk.Button(self, text="识别教务处验证码", command=self.predict_distance_verifycode)
        self.predict_distance_verifycode.place(x=190, y=165)

    def do_train(self):
        self.CNN.start_train()

    def do_predict(self):
        result = self.CNN.predict_validation()
        self.result_display.delete(0, tk.END)
        self.result_display.insert(0, '识别结果：' + result)

    def place_verifycode(self):
        byte_image = self.Schedule.get_verify_code()
        self.new_pilImage1 = self.Img.byte2jpeg(byte_image)
        self.new_tkImage1 = ImageTk.PhotoImage(image=self.new_pilImage1)
        self.new_pilImage2 = self.Img.process(byte_image)
        self.new_tkImage2 = ImageTk.PhotoImage(image=self.new_pilImage2)
        self.distance_verifycode1 = tk.Label(self, image=self.new_tkImage1)
        self.distance_verifycode1.place(x=100, y=200)
        self.distance_verifycode2 = tk.Label(self, image=self.new_tkImage2)
        self.distance_verifycode2.place(x=180, y=200)

    def predict_distance_verifycode(self):
        result = self.CNN.predict(self.new_pilImage2)
        print(result)
        self.verifycode_display.delete(0, tk.END)
        self.verifycode_display.insert(0, '识别结果：' + result)


if __name__ == "__main__":
    windows = GUI()
    windows.mainloop()
