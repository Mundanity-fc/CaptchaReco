import requests
import re
import yaml


class ScheduleClass:
    def __init__(self):
        config_file = open(r'config/account.yaml', 'r', encoding='utf-8')
        config_content = config_file.read()
        self.config = yaml.load(config_content, Loader=yaml.FullLoader)
        self.username = self.config['username']
        self.password = self.config['password']
        self.get_login_cookie()
        self.cookie = ""

    def get_login_cookie(self):
        # 初次发送GET请求
        first_request = requests.get("http://202.119.81.113:8080/")
        # 获取访问的cookie
        self.login_cookie = first_request.headers.get('Set-Cookie')[11:43]
        self.login_cookie = {'JSESSIONID': self.login_cookie}
        return self.login_cookie

    def get_verify_code(self):
        # 以获取的cookie获取新的验证码
        image = requests.get("http://202.119.81.113:8080/verifycode.servlet", cookies=self.login_cookie)
        return image.content

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
