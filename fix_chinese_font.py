"""
中文字体配置工具
解决matplotlib中文显示问题
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import os


def detect_chinese_fonts():
    """检测系统中可用的中文字体"""
    print("\n" + "="*60)
    print("检测系统中文字体")
    print("="*60)
    
    system = platform.system()
    print(f"操作系统: {system}")
    
    # 获取所有字体
    all_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 中文字体关键词
    chinese_keywords = [
        'SimHei', 'SimSun', 'Microsoft', 'KaiTi', 'FangSong',  # Windows
        'PingFang', 'Heiti', 'STHeiti', 'STSong', 'Arial Unicode',  # macOS
        'WenQuanYi', 'Noto', 'Droid Sans', 'AR PL'  # Linux
    ]
    
    # 查找中文字体
    chinese_fonts = []
    for font in all_fonts:
        for keyword in chinese_keywords:
            if keyword.lower() in font.lower():
                chinese_fonts.append(font)
                break
    
    if chinese_fonts:
        print(f"\n找到 {len(chinese_fonts)} 个中文字体:")
        for i, font in enumerate(chinese_fonts, 1):
            print(f"  {i}. {font}")
        return chinese_fonts
    else:
        print("\n❌ 未找到中文字体！")
        return []


def recommend_fonts():
    """推荐字体配置"""
    system = platform.system()
    
    print("\n" + "="*60)
    print("推荐字体配置")
    print("="*60)
    
    if system == 'Windows':
        print("\nWindows系统推荐:")
        print("  1. SimHei (黑体) - 推荐")
        print("  2. Microsoft YaHei (微软雅黑)")
        print("  3. SimSun (宋体)")
        print("  4. KaiTi (楷体)")
        
        print("\n配置代码:")
        print("  plt.rcParams['font.sans-serif'] = ['SimHei']")
        
        recommended = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi']
        
    elif system == 'Darwin':
        print("\nmacOS系统推荐:")
        print("  1. Songti SC (宋体) - 推荐")
        print("  2. Heiti TC (黑体)")
        print("  3. Arial Unicode MS")
        print("  4. Hiragino Sans GB")
        
        print("\n配置代码:")
        print("  plt.rcParams['font.sans-serif'] = ['Songti SC']")
        
        recommended = ['Songti SC', 'Heiti TC', 'Arial Unicode MS', 'Hiragino Sans GB']
        
    else:
        print("\nLinux系统推荐:")
        print("  1. WenQuanYi Micro Hei (文泉驿微米黑) - 推荐")
        print("  2. WenQuanYi Zen Hei (文泉驿正黑)")
        print("  3. Noto Sans CJK SC")
        print("  4. Droid Sans Fallback")
        
        print("\n安装命令:")
        print("  Ubuntu/Debian:")
        print("    sudo apt-get install fonts-wqy-microhei")
        print("    sudo apt-get install fonts-wqy-zenhei")
        print("  CentOS/RHEL:")
        print("    sudo yum install wqy-microhei-fonts")
        
        print("\n配置代码:")
        print("  plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']")
        
        recommended = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'Droid Sans Fallback']
    
    return recommended


def configure_chinese_fonts(font_name=None):
    """
    配置中文字体
    
    Args:
        font_name: 指定字体名称，如果为None则自动选择
    """
    print("\n" + "="*60)
    print("配置中文字体")
    print("="*60)
    
    if font_name:
        # 使用指定字体
        plt.rcParams['font.sans-serif'] = [font_name]
        print(f"✓ 已设置字体: {font_name}")
    else:
        # 自动选择
        system = platform.system()
        
        if system == 'Windows':
            fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi']
        elif system == 'Darwin':
            # 使用经过验证的字体配置
            fonts = ['Songti SC', 'Heiti TC', 'Arial Unicode MS', 'Hiragino Sans GB']
            # 显式加载字体文件
            try:
                font_paths = [
                    '/System/Library/Fonts/Supplemental/Songti.ttc',
                    '/System/Library/Fonts/STHeiti Medium.ttc'
                ]
                for font_path in font_paths:
                    if os.path.exists(font_path):
                        fm.fontManager.addfont(font_path)
            except:
                pass
        else:
            fonts = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'Droid Sans Fallback', 'DejaVu Sans']
        
        plt.rcParams['font.sans-serif'] = fonts
        print(f"✓ 已设置字体列表: {fonts}")
    
    # 设置负号正常显示
    plt.rcParams['axes.unicode_minus'] = False
    print("✓ 已设置负号正常显示")
    
    # 清除字体缓存
    try:
        cache_dir = fm.get_cachedir()
        cache_file = os.path.join(cache_dir, 'fontlist-v330.json')
        if os.path.exists(cache_file):
            print(f"\n字体缓存位置: {cache_file}")
            print("如果中文仍不显示，可以删除此文件后重启Python")
    except:
        pass


def test_chinese_display():
    """测试中文显示"""
    print("\n" + "="*60)
    print("测试中文显示")
    print("="*60)
    
    import numpy as np
    
    # 创建测试图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('中文字体测试 - Chinese Font Test', fontsize=16, fontweight='bold')
    
    # 测试1: 折线图
    ax1 = axes[0, 0]
    x = np.arange(10)
    y = np.random.randn(10).cumsum()
    ax1.plot(x, y, 'o-', linewidth=2, markersize=8)
    ax1.set_title('折线图测试')
    ax1.set_xlabel('时间 (天)')
    ax1.set_ylabel('收益率 (%)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(['策略收益'], loc='best')
    
    # 测试2: 柱状图
    ax2 = axes[0, 1]
    categories = ['估值', '成长', '盈利', '质量']
    values = [0.25, 0.30, 0.20, 0.25]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    ax2.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_title('因子权重分布')
    ax2.set_ylabel('权重')
    ax2.set_ylim(0, 0.4)
    for i, v in enumerate(values):
        ax2.text(i, v + 0.01, f'{v:.0%}', ha='center', va='bottom')
    
    # 测试3: 饼图
    ax3 = axes[1, 0]
    labels = ['金融', '科技', '医药', '消费', '其他']
    sizes = [25, 30, 15, 20, 10]
    explode = (0.1, 0, 0, 0, 0)
    ax3.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90, colors=colors)
    ax3.set_title('行业配置')
    
    # 测试4: 文本显示
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    text_content = """测试内容 Test Content:

✓ 简体中文: 你好世界
✓ 标点符号: ，。！？；：
✓ 数字: 0123456789
✓ 英文: Hello World
✓ 混合: 收益率=15.5%
✓ 特殊字符: +-×÷≈≠

当前字体设置:
""" + str(plt.rcParams['font.sans-serif'][:3])
    
    # 不使用 monospace，使用默认字体（支持中文）
    ax4.text(0.1, 0.5, text_content, fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # 保存测试图
    output_path = 'chinese_font_test.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 测试图已保存: {output_path}")
    
    plt.show()
    
    print("\n如果图表中的中文显示正常，说明配置成功！")
    print("如果显示为方框□，请尝试:")
    print("  1. 安装中文字体")
    print("  2. 删除matplotlib字体缓存")
    print("  3. 重启Python")


def fix_chinese_font():
    """一键修复中文字体问题"""
    print("\n" + "="*60)
    print("中文字体一键修复")
    print("="*60)
    
    # 1. 检测字体
    chinese_fonts = detect_chinese_fonts()
    
    # 2. 推荐配置
    recommended = recommend_fonts()
    
    # 3. 自动配置
    configure_chinese_fonts()
    
    # 4. 测试显示
    try:
        test_chinese_display()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        print("\n可能的解决方案:")
        print("  1. 确保已安装中文字体")
        print("  2. 重启Python环境")
        print("  3. 手动设置字体")


def manual_font_config():
    """手动配置字体指南"""
    print("\n" + "="*60)
    print("手动配置指南")
    print("="*60)
    
    print("""
方案1: 在代码中设置（推荐）
---------------------------------
import matplotlib.pyplot as plt

# Windows
plt.rcParams['font.sans-serif'] = ['SimHei']

# macOS  
plt.rcParams['font.sans-serif'] = ['PingFang SC']

# Linux
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']

# 负号正常显示
plt.rcParams['axes.unicode_minus'] = False


方案2: 修改配置文件（永久生效）
---------------------------------
1. 查找配置文件位置:
   import matplotlib
   print(matplotlib.matplotlib_fname())

2. 编辑 matplotlibrc 文件:
   找到这两行并修改:
   font.sans-serif: SimHei, Microsoft YaHei, ...
   axes.unicode_minus: False

3. 删除字体缓存:
   import matplotlib.font_manager as fm
   print(fm.get_cachedir())
   # 删除该目录下的 fontlist-v330.json

4. 重启Python


方案3: 使用FontProperties（临时方案）
---------------------------------
from matplotlib.font_manager import FontProperties

# 指定字体
font = FontProperties(fname='/path/to/font.ttf', size=12)

# 使用字体
plt.title('标题', fontproperties=font)
plt.xlabel('X轴', fontproperties=font)


方案4: 安装字体（Linux）
---------------------------------
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install fonts-wqy-microhei fonts-wqy-zenhei

# CentOS/RHEL
sudo yum install wqy-microhei-fonts wqy-zenhei-fonts

# Arch Linux
sudo pacman -S wqy-microhei wqy-zenhei

# 刷新字体缓存
fc-cache -fv


常见问题排查
---------------------------------
Q1: 设置后仍显示方框？
A: 删除matplotlib缓存，重启Python

Q2: 找不到中文字体？
A: 运行 detect_chinese_fonts() 查看可用字体

Q3: Linux下没有中文字体？
A: 安装 fonts-wqy-microhei 包

Q4: 部分中文显示，部分不显示？
A: 字体不全，尝试多个字体: ['Font1', 'Font2', 'Font3']
""")


def show_font_list():
    """显示所有可用字体"""
    print("\n" + "="*60)
    print("系统所有字体列表")
    print("="*60)
    
    fonts = sorted([f.name for f in fm.fontManager.ttflist])
    
    print(f"\n共找到 {len(fonts)} 个字体:\n")
    
    # 分列显示
    for i in range(0, len(fonts), 3):
        row = fonts[i:i+3]
        print(f"  {row[0]:30s}", end="")
        if len(row) > 1:
            print(f"  {row[1]:30s}", end="")
        if len(row) > 2:
            print(f"  {row[2]:30s}", end="")
        print()
    
    # 保存到文件
    with open('system_fonts.txt', 'w', encoding='utf-8') as f:
        for font in fonts:
            f.write(font + '\n')
    print(f"\n✓ 字体列表已保存到: system_fonts.txt")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'detect':
            detect_chinese_fonts()
        elif command == 'recommend':
            recommend_fonts()
        elif command == 'configure':
            font = sys.argv[2] if len(sys.argv) > 2 else None
            configure_chinese_fonts(font)
        elif command == 'test':
            configure_chinese_fonts()
            test_chinese_display()
        elif command == 'manual':
            manual_font_config()
        elif command == 'list':
            show_font_list()
        else:
            print("未知命令")
    else:
        # 默认执行一键修复
        fix_chinese_font()
