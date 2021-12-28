import requests
import re
import yaml


class ScheduleClass:
    def __init__(self):
        # 获取学号及密码
        config_file = open(r'config/account.yaml', 'r', encoding='utf-8')
        config_content = config_file.read()
        self.config = yaml.load(config_content, Loader=yaml.FullLoader)
        self.username = self.config['username']
        self.password = self.config['password']
        self.get_login_cookie()
        self.cookie = ""
        self.rank = []
        self.schedule = []

    # 获取第一次cookie（匹配验证码提交cookie）
    def get_login_cookie(self):
        # 初次发送GET请求
        first_request = requests.get("http://202.119.81.113:8080/")
        # 获取访问的cookie
        self.login_cookie = first_request.headers.get('Set-Cookie')[11:43]
        self.login_cookie = {'JSESSIONID': self.login_cookie}
        return self.login_cookie

    # 获取新的验证码
    def get_verify_code(self):
        # 以获取的cookie获取新的验证码
        image = requests.get("http://202.119.81.113:8080/verifycode.servlet", cookies=self.login_cookie)
        return image.content

    # 获取第二次cookie（教务系统页面cookie）
    def login(self, verifycode):
        # 合成登录表单
        useDogeCode = ""
        login_form = {'USERNAME': self.username, 'PASSWORD': self.password, 'useDogeCode': useDogeCode,
                      'RANDOMCODE': verifycode}
        # 发送POST请求
        login = requests.post("http://202.119.81.113:8080/Logon.do?method=logon", data=login_form,
                              cookies=self.login_cookie)
        # 获取新的cookie
        cookie_pair = login.history[1].headers.get('Set-Cookie')[11:43]
        self.cookie = {'JSESSIONID': cookie_pair}
        return self.cookie

    # 成绩查询
    def get_rank(self):
        kksj = ''
        kcxz = ''
        kcmc = ''
        xsfs = 'all'
        # 根据页面的元素发送post请求
        # kksj：开课时间，kcxz：课程性质，kcmc：课程名称，xsfs：显示方式
        # 具体参数类型参照教务处html元素
        rank_data = {'kksj': kksj, 'kcxz': kcxz, 'kcmc': kcmc, 'xsfs': xsfs}
        # 发送post请求
        get_rank = requests.get('http://202.119.81.112:9080/njlgdx/kscj/cjcx_list', data=rank_data, cookies=self.cookie)
        get_rank.encoding = 'utf-8'
        html = get_rank.text

        # 获取表格内容
        table = re.findall(r'<table(.*?)</table>', html, re.S)
        rank_list = re.findall(r'<tr>(.*?)</tr>', table[1], re.S)
        # 移除表头内容
        rank_list.pop(0)
        # 返回的数据集
        data = []
        # 截取每行内容
        for i in range(len(rank_list)):
            data.append(re.findall(r'<td(.*?)</td>', rank_list[i], re.S))
        # 删除内容的css样式残余
        for i in range(len(data)):
            for j in range(len(data[i])):
                str_list = data[i][j].split('>')
                data[i][j] = str_list[1]
        self.rank = data
        return self.rank
