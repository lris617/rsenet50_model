import matplotlib.pyplot as plt
import os
from matplotlib import font_manager

# 动态注册Arial字体
font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "arial-regular.ttf")
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Arial'
else:
    plt.rcParams['font.family'] = 'DejaVu Sans'
    print("警告：未找到arial-regular.ttf，继续使用默认字体。")

from matplotlib.font_manager import FontProperties

plt.figure()
if os.path.exists(font_path):
    font_prop = FontProperties(fname=font_path)
else:
    font_prop = None

plt.title("Test Arial 字体", fontsize=16, fontproperties=font_prop)
plt.xlabel("X轴", fontsize=14, fontproperties=font_prop)
plt.ylabel("Y轴", fontsize=14, fontproperties=font_prop)
plt.plot([0, 1, 2], [1, 2, 3], label="示例线")
plt.legend(prop=font_prop)
plt.tight_layout()
plt.savefig("test_font_output.png", dpi=100)
plt.close()
print("测试图片已保存为 test_font_output.png")
