# curated list of useful commands when working with Cluster

## Slurm 

* **sinfo** shows the status of ressources
* **sfree** lists the free resources
* **sblock --partition=Partition -c=NumOfCPUs --mem=MemoryG --time=hh::mm::ss** reserves resources of a given partition
* **squeue** shows all running jobs (takes some time to update)
* **sjob** shows the jobs started by oneself
* **scancel jobid** ends the job
* **sbatch** starts a job via a shell-file

## tmux
* **tmux new -s SessionName** creates a new tmux Session
* **tmux attach -t SessionName** attaches to a running tmux Session
* **tmux ls** lists all running sessions
* **tmux kill-session -t SessionName** ends a session
* **Ctrl-B D** detaches from the currently viewed session
