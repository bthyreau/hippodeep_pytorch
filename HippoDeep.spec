# -*- mode: python -*-
a = Analysis(['model_apply_head_and_hippo.py'],
             excludes=['lib2to3', 'win32com', 'win32pdh','win32pipe','PIL'],
             hookspath=None,
             runtime_hooks=None)
             
#avoid warning
for d in a.datas:
    if '_C.cp37-win_amd64.pyd' in d[0]: 
        a.datas.remove(d)
        break

#manually include model /torchparams
a.datas += Tree('./torchparams', prefix='./torchparams') 

#remove unnecessary stuff        
a.datas = [x for x in a.datas if not ('tk8.6\msgs' in os.path.dirname(x[1]))]            
a.datas = [x for x in a.datas if not ('tk8.6\images' in os.path.dirname(x[1]))]            
a.datas = [x for x in a.datas if not ('tk8.6\demos' in os.path.dirname(x[1]))]            
a.datas = [x for x in a.datas if not ('tcl8.6\opt0.4' in os.path.dirname(x[1]))]            
a.datas = [x for x in a.datas if not ('tcl8.6\http1.0' in os.path.dirname(x[1]))]            
a.datas = [x for x in a.datas if not ('tcl8.6\encoding' in os.path.dirname(x[1]))]            
a.datas = [x for x in a.datas if not ('tcl8.6\msgs' in os.path.dirname(x[1]))]            
a.datas = [x for x in a.datas if not ('tcl8.6\tzdata' in os.path.dirname(x[1]))]

pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='HippoDeep.exe',
          debug=False,
          strip=None,
          upx=False,
          console=True)         
        