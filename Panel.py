import tkinter as tk
import tkinter.messagebox
from ModelClass import ModelClass
from ScheduleClass import ScheduleClass
from ImageProcess import ImageProcess
from PIL import Image, ImageTk


class Panel(tkinter.Tk):
    """
    基本的GUI界面，能够完成模型训练和图像的检测
    """""
    def __init__(self):
        super().__init__()
        self.filename = None
        self.title("教务处信息获取")
        self.geometry("380x400")
        self.CNN = ModelClass()
        self.Schedule = ScheduleClass()
        self.Img = ImageProcess()
        self.layout()
        self.verifycode = ""

    def layout(self):
        """
        进行GUI的布局设计
        :return: 无返回值
        """""
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
        self.get_distance_verifycode = tk.Button(self, text="①获取教务处验证码", command=self.place_verifycode)
        self.get_distance_verifycode.place(x=10, y=165)
        self.predict_distance_verifycode = tk.Button(self, text="②识别教务处验证码", command=self.predict_distance_verifycode)
        self.predict_distance_verifycode.place(x=130, y=165)
        self.get_schedule_info = tk.Button(self, text="③获取最新考试成绩", command=self.print_rank)
        self.get_schedule_info.place(x=250, y=165)

    # 开始进行模型训练
    def do_train(self):
        """
        调用模型训练函数
        :return: 无返回值
        """""
        self.CNN.start_train()

    # 进行本地文件的预测并输出其结果至文本框
    def do_predict(self):
        """
        调用模型识别本地图片内容，结果输出到第一个文本框
        :return: 无返回值
        """""
        result = self.CNN.predict_validation()
        self.result_display.delete(0, tk.END)
        self.result_display.insert(0, '识别结果：' + result)

    # 获取教务处的验证码并原图片和处理后的图片
    def place_verifycode(self):
        """
        获取教务处的验证码并进行处理
        将原图片和处理后的图片显示于按钮下方
        :return: 无返回值
        """""
        byte_image = self.Schedule.get_verify_code()
        self.new_pilImage1 = self.Img.byte2jpeg(byte_image)
        self.new_tkImage1 = ImageTk.PhotoImage(image=self.new_pilImage1)
        self.new_pilImage2 = self.Img.process(byte_image)
        self.new_tkImage2 = ImageTk.PhotoImage(image=self.new_pilImage2)
        self.distance_verifycode1 = tk.Label(self, image=self.new_tkImage1)
        self.distance_verifycode1.place(x=100, y=200)
        self.distance_verifycode2 = tk.Label(self, image=self.new_tkImage2)
        self.distance_verifycode2.place(x=180, y=200)

    # 识别获取到的验证码
    def predict_distance_verifycode(self):
        """
        识别获取到的验证码的内容
        :return: 无返回值
        """""
        result = self.CNN.predict(self.new_pilImage2)
        print(result)
        self.verifycode_display.delete(0, tk.END)
        self.verifycode_display.insert(0, '识别结果：' + result)
        self.verifycode = result

    # 弹框输出最新的成绩
    def print_rank(self):
        """
        以弹窗形式显示最新的成绩结果
        :return: 无返回值
        """""
        # 完成登录操作
        self.Schedule.login(self.verifycode)
        ranklist = self.Schedule.get_rank()
        # 判断是否获取到成绩
        if len(ranklist) > 1:
            last = ranklist.pop()
            tkinter.messagebox.showinfo('最新科目成绩！！', '最新科目为：%s\n成绩为：%s'%(last[3], last[4]))
        else:
            # 未获取到成绩时弹窗提示
            tkinter.messagebox.showinfo('出错了', ranklist[0])