from nonebot import on_command, CommandSession
from ModelClass import ModelClass
from ScheduleClass import ScheduleClass
from ImageProcess import ImageProcess


@on_command('查成绩')
async def get_rank(session: CommandSession):
    schedule = ScheduleClass()
    process = ImageProcess()
    model = ModelClass()
    img = schedule.get_verify_code()
    processed = process.process(img)
    verifycode = model.predict(processed)
    schedule.login(verifycode)
    ranklist = schedule.get_rank()
    if len(ranklist) > 1:
        last = ranklist.pop()
        message = '最新科目为：%s\n成绩为：%s' % (last[3], last[4])
    else:
        message = "请重试！"
    await session.send(message)
