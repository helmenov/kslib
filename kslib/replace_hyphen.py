import re

def replace_hyphen(text, replace_out):
    """全ての横棒を半角ハイフンに置換する
    Args:
        text (str): 入力するテキスト
        replace_out (str): 置換したい文字列
    Returns:
        (str): 置換後のテキスト
    """
    hyphens = '-˗ᅳ᭸‐‑‒–—―⁃⁻−▬─━➖ーㅡ﹘﹣－ｰ𐄐𐆑 '
    hyphens = '|'.join(hyphens)
    return re.sub(hyphens, replace_out, text)


hyphendict = {
    "Hyphen-Minus":45, 
    "Modifier Letter Minus Sign":727, 
    "Blinese Musical Symbol Left-HandOpen Pang":7032, 
    "Hyphen (hyphen)":8208,
    "Non-Breaking Hyphen":8209, 
    "Figure Dash":8210, 
    "En Dash (ndash)": 8211, 
    "Em Dash (mdash)": 8212,
    "Horizontal Bar (horbar)": 8213, 
    "Hyphen Bullet (hybull)": 8259, 
    "Superscript Minus": 8315, 
    "Minus Sign": 8722, 
    "Black Rectangle": 9644, 
    "Box Drawings Light Horizontal": 9472, 
    "Heavy Minus Sign": 9473, 
    "Box Drawings Heavy Horizontal": 10134, 
    "Katakana-Hiragana Prolonged Sound Mark": 12540, 
    "Hangul Letter Eu": 12641, 
    "Small Em Dash": 65112, 
    "Small Hyphen-Minus": 65123, 
    "Fullwidth Hyphen-Minus": 65293, 
    "Halfwidth Katakana-Hiragana Prolonged Sound Mark": 65392,
    "Aegean Number Ten": 65808, 
    "Roman Uncia Sign": 65937, 
    "Ogham Space Mark": 5760 
}

def check_hyphen(text:str):
    for char in text:
        upnt = ord(char)
        if upnt in hyphendict.values():
            keys = [k for k, v in hyphendict.items() if v == upnt]
            print(f"{char}:{keys}")
        else:
            print(f"{char}")

