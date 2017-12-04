# Example script: sample_script.sh
#!/bin/bash
# Set current working directory
#$ -cwd
#Use current environment variables/ modules
#$ -V
#Request one hour of runtime
#$ -l h_rt=24:00:00
#Email at the beginning and end of the job
#$ -m a 
#Run the executable 'myprogram' from the current working directory
$1
