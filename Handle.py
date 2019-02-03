# 详细操作： 图片处理、行为运行、状态判断等

import time,os
import numpy as np
from PIL import ImageGrab,Image

class Handle:
    def __init__(self):
        self.debug = False
        self.box = (3,43,483,823)           # 完整截图区域
        self.box_view = (0, 120, 460, 580)  # 神经网络的视觉区域
        self.box_reward = (144, 20, 334, 90)# 比分板区域
        self.image = None                   # 未初始化的截图
        self.image_view = None              # 未初始化的神经网络视野
        self.image_reward = None            # 未初始化的分数板区域
        self.state = []                     # 初始化状态
        self.score_previous = 0             # 之前的分数

    def getimage(self):
        self.image = ImageGrab.grab(self.box).convert('L')
        self.image_view = self.image.crop(self.box_view)
        self.image_reward = self.image.crop(self.box_reward)

        if self.debug:
            self.image.save('./log/screen.png')
            self.image_view.save(f'./log/state.png')
            self.image_reward.save('./log/reword.png')

    def getstate(self):
        self.state = []
        for _ in range(4):
            time.sleep(0.06)
            self.getimage()
            self.state.append(self.image_view)
        self.state = np.stack(self.state,axis=0)

    def action(self,action):
        score = self.getscore()

        judge = self.judge()

        if score is not None and judge!=0:
            if action==0:
                os.system('nox_adb.exe shell input tap 130 640')
                print('选左边')
            else:
                os.system('nox_adb.exe shell input tap 360 640')  # 选右边
                print('选右边')
            reward = float(judge)*0.6 + int(score)*0.4
            print(f'score:{score},judge:{judge},reward:{reward}')
            self.getstate()                                       # 下一个状态
            return self.state,reward,0
        else:
            return None,None,1

    def getscore(self):
        image_one = self.image_reward.crop((0,0,72,70))
        for _,_,filename in os.walk('./source/one'):
            for file in filename:
                name = os.path.join('./source/one',file)
                image = Image.open(name)
                if self.similar(image_one,image)>90:
                    return int(file.replace('.png',''))
        print('No matching')
        return None


    def judge(self):
        if self.similar(self.image.crop((150, 570, 330, 630)), Image.open('./source/restart.png')) > 90:
            os.system('nox_adb.exe shell input tap 240 600')  # 重新开始
            print('重新开始')
            return 0  # 无操作
        if self.similar(self.image.crop((150, 620, 330, 670)), Image.open('./source/loss_continue.png')) > 90:
            os.system('nox_adb.exe shell input tap 80 623')  # 丢分点击
            print('失分惩罚')
            return -2  # 惩罚
        if self.similar(self.image.crop((150, 620, 330, 670)), Image.open('./source/get_continue.png')) > 90:
            os.system('nox_adb.exe shell input tap 240 400')  # 得分点击
            print('得分奖励')
            return 2  # 奖励
        # if self.similar(self.image.crop((110,620,390,665)), Image.open('./source/wait.png')) > 90:
        #     os.system('nox_adb.exe shell input tap 240 400')  # 得分点击,对战模式
        #     print('得分奖励')
        #     return 2  # 奖励
        if self.similar(self.image.crop((170,590,360,620)), Image.open('./source/nextoppent.png'))>90:
            os.system('nox_adb.exe shell input tap 250 610')  # 点击挑战下一个对手
            print('挑战下一个对手')
            return 4  # 奖励
        if self.similar(self.image.crop((170,675,310,705)), Image.open('./source/skip.png'))>90:
            os.system('nox_adb.exe shell input tap 230 700')  # 跳过看视频复活
            print('跳过看视频复活')
            time.sleep(0.5)
            return 0  # 无操作
        if self.similar(self.image.crop((170,640,310,670)), Image.open('./source/video_skip.png'))>90:
            os.system('nox_adb.exe shell input tap 235 660')  # 跳过看视频奖励
            print('跳过看视频奖励')
            return 0  # 无操作
        if self.similar(self.image.crop((420,250,470,300)), Image.open('./source/x.png'))>90:
            os.system('nox_adb.exe shell input tap 445 285')  # 关闭向朋友推荐
            print('关闭向朋友推荐')
            return 0  # 无操作

        if self.similar(self.image.crop((34,560,90,615)), Image.open('./source/rechallenge.png'))>90:
            os.system('nox_adb.exe shell input tap 60 600')   # 重新挑战
            print('重新挑战')
            return 0  # 无操作
        return 0.1



    @staticmethod
    def similar(image_1,image_2):
        lh, rh = image_1.histogram(), image_2.histogram()
        ret = 100 * sum(1 - (0 if l == r else float(abs(l - r)) / max(l, r)) for l, r in zip(lh, rh)) / len(lh)
        return ret




if __name__=='__main__':
    import time
    operate = Handle()
    operate.getstate()

    for i in range(400):
        time.sleep(1)
        operate.getimage()
        operate.judge()




