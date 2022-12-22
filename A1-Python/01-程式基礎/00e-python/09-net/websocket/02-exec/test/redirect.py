import sys
import io

old_stdout = sys.stdout
sys.stdout = mystdout = io.StringIO()

exec(f'x,y=1,2\nz=x+y\nprint("aaa")\nprint(x,y,z)\n')
mystdout.flush()

out = mystdout.getvalue()
mystdout.close()
sys.stdout = old_stdout
print('out=', out)
