import subprocess, os
from fontTools.cffLib import CFFFontSet
from fontTools.ttLib import newTable
from fontTools.fontBuilder import FontBuilder
from fontTools.agl import toUnicode
from fontTools.afmLib import AFM
from fontTools.feaLib.builder import addOpenTypeFeaturesFromString
os.makedirs('fonts', exist_ok=True)
for base, style, mac in [('bchr8a','Regular',0),('bchri8a','Italic',2),('bchb8a','Bold',1),('bchbi8a','BoldItalic',3)]:
    pfb = subprocess.run(['kpsewhich', base+'.pfb'], capture_output=True, text=True).stdout.strip()
    afm = AFM(subprocess.run(['kpsewhich', base+'.afm'], capture_output=True, text=True).stdout.strip())
    subprocess.run(['tx','-cff','+b','-Z','-std',pfb,'tmp.cff'], check=True, capture_output=True)
    cff = CFFFontSet(); cff.decompile(open('tmp.cff','rb'), None)
    order = cff[cff.fontNames[0]].getGlyphOrder()
    fb = FontBuilder(1000, isTTF=False); fb.setupGlyphOrder(order)
    cmap = {}
    for g in order:
        u = toUnicode(g)
        if len(u) == 1: cmap.setdefault(ord(u), g)
    fb.setupCharacterMap(cmap)
    fb.setupHorizontalMetrics({g: (afm._chars[g][1], afm._chars[g][2][0]) if g in afm._chars else (0, 0) for g in order})
    fb.setupHorizontalHeader(ascent=afm.Ascender, descent=afm.Descender)
    ps = 'Charter-' + style; cff.fontNames = [ps]
    fb.setupNameTable({'familyName': 'Charter', 'styleName': style.replace('BoldItalic','Bold Italic'), 'psName': ps})
    fb.setupOS2(sTypoAscender=afm.Ascender, sTypoDescender=afm.Descender, usWinAscent=afm.Ascender,
                usWinDescent=-afm.Descender, fsType=0, usWeightClass=700 if 'Bold' in style else 400,
                fsSelection={'Regular':0x40,'Italic':0x01,'Bold':0x20,'BoldItalic':0x21}[style])
    fb.setupPost(italicAngle=-10 if 'Italic' in style else 0)
    fb.font['CFF '] = newTable('CFF '); fb.font['CFF '].cff = cff
    fb.font['head'].macStyle = mac; fb.font.sfntVersion = 'OTTO'
    kern = '\n'.join(f'pos {l} {r} {v};' for (l, r), v in afm._kerning.items())
    addOpenTypeFeaturesFromString(fb.font, 'languagesystem DFLT dflt;\nlanguagesystem latn dflt;\n'
        'feature liga { sub f i by fi; sub f l by fl; } liga;\n' f'feature kern {{\n{kern}\n}} kern;\n')
    fb.save(f'fonts/Charter-{style}.otf')
os.remove('tmp.cff')
