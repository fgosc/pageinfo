import subprocess
from pathlib import Path

root = Path('./tests/images')
entries = [(e.parent.name, str(e)) for e in root.glob('**/*.*')]

for parent, file in entries:
    cmd = f'./pageinfo.py qp -ds -dp {parent}- {file}'
    print(cmd)
    subprocess.run(cmd, shell=True)

