"""sslm_verify must refuse a SchemaMasks section the runtime loader refuses (array S2 finding, 2026-10-25).

Attaches G1 to the converter-v2 synthetic base, then corrupts one SCM1 word at a time and stops at the
first corruption the verifier refuses. Before the fix, the verifier accepted offset 16 (SchemaMasksOutOfBounds
at map time). Usage: python3 tests/array-s2/verify_schema_masks_check.py [path/to/sslm_verify]
"""
import sys, subprocess, json, struct
from pathlib import Path
R=Path(__file__).resolve().parents[2]; C=R/'tests/t3094-multicommand-s0/converter-v2'
sys.path[:0]=[str(C),str(R/'tools'),str(R/'tools/reference_pipeline')]
import cv2_attach_fixtures as A, sslm_format as SF
import tempfile; w=Path(tempfile.mkdtemp()); fx=A.build(w); print(sorted(fx))
base=fx.get('base') or fx[sorted(k for k in fx if 'base' in k)[0]]
out=w/'g1.sslm'
r=subprocess.run([sys.executable,str(R/'tools/sslm_attach_schema.py'),'--base',str(base),'--ckpt',str(A.F.TOKENIZER_DIR),'--schema','intent_gate:'+str(R/'tests/t3094-multicommand-s0/v2/g1.schema.json'),'--out',str(out),'--force'],capture_output=True,text=True); print(r.returncode, r.stdout[-200:], r.stderr[-300:])
h=SF.read_header(str(out)); s=[x for x in h['sections'] if x['type']==SF.SectionType.SCHEMA_MASKS][0]
secs,flags=A._sections_of(SF,out)
sc=[x for x in secs if x.type==SF.SectionType.SCHEMA_MASKS][0]
from dataclasses import replace as _r
for o in range(16,256,4):
    d=bytearray(sc.data); struct.pack_into('<I',d,o,struct.unpack_from('<I',d,o)[0]+4)
    bad=w/'bad.sslm'; SF.write_artifact(str(bad),[(_r(x,data=bytes(d)) if x is sc else x) for x in secs],flags)
    res=[]
    for binp in (Path(sys.argv[1]) if len(sys.argv)>1 else R/'build/sslm_verify',):
        p=subprocess.run([str(binp),str(bad),str(w/'m.json')],capture_output=True,text=True)
        res.append((p.returncode,(p.stdout+p.stderr).strip()[:150]))
    if res[0][0]!=0:
        print('refused at offset',o,res[0]); break
else:
    sys.exit('FAIL: no corruption in the first 256 bytes was refused')
