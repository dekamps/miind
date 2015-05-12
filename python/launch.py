import subprocess 

f=open('joblist')
lines=f.readlines()
for line in lines:
	name=line.strip()
	subprocess.call(['qsub','./sub.sh',name])
