import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import urllib.request
import warnings

def test():
    font_dir = "fonts"
    os.makedirs(font_dir, exist_ok=True)
    font_path = os.path.join(font_dir, "NotoSansDevanagari-Regular.ttf")
    if not os.path.exists(font_path):
        url = "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Regular.ttf"
        urllib.request.urlretrieve(url, font_path)
    
    fm.fontManager.addfont(font_path)
    prop = fm.FontProperties(fname=font_path)
    print(f"Added font family: {prop.get_name()}")
    
    plt.rcParams['font.sans-serif'] = [prop.get_name(), 'sans-serif']
    
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, "English Text and Hindi: भ", fontsize=12)
    plt.savefig("test.png")
    print("Saved test.png")

if __name__ == "__main__":
    test()
