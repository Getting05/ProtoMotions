"""List selected motion IDs/names for human category/group review (CPU only)."""
import argparse
import json
from pathlib import Path
import re
import torch
from protomotions.state_estimation.data import canonical_motion_group

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--motion-file',required=True); p.add_argument('--output',required=True)
p.add_argument('--include',default='.',help='Regex on motion file names')
p.add_argument('--limit',type=int,default=100)
p.add_argument('--category',default='unclassified',help='Assign a known category after inspecting selected clips')
a=p.parse_args()
blob=torch.load(a.motion_file,map_location='cpu',weights_only=False,mmap=True)
names=blob['motion_files']; selected=[]
for i,name in enumerate(names):
    name=str(name)
    if re.search(a.include,name,re.IGNORECASE):
        selected.append(dict(motion_id=i,motion_name=name,group_id=canonical_motion_group(name),category=a.category))
        if len(selected)>=a.limit: break
if not selected: raise ValueError('No matching reference motions')
output=Path(a.output)
if output.exists(): raise FileExistsError(output)
output.write_text(json.dumps(dict(motions=selected),indent=2))
print(f'Wrote {len(selected)} candidates. Review category and shared group_id for mirrored/cropped variants before collection.')
