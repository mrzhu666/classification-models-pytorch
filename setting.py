import platform
import yaml
import os

print()
print('配置文件加载')
print('load config')

# 如果是 python dir/setting.py 直接运行的话，不需要绝对路径也行
curDir=os.path.dirname(__file__)  # 当前文件的文件夹绝对路径
if platform.system().lower() == 'windows':
    print("windows")
    f=open(curDir+'/config.yaml', encoding='utf-8')
    config=yaml.safe_load(f)
elif platform.system().lower() == 'linux':
    print("linux")
    f=open(curDir+'/config.yaml', encoding='utf-8')
    config=yaml.safe_load(f)
    result=os.popen('echo "$USER"')
    user=result.read().strip()  # 获取用户名
    config.update(config['user'][user])

# config['server_path']=config['user'][user]['server_path']
print(config)
print()