#!/bin/bash 
#
#all commands that start with SBATCH contain commands that are just used by SLURM for scheduling
#################
#partition name
#SBATCH --partition=napoli-gpu
#################
#number of GPUs
##SBATCH --gres=gpu:0
#################
#set a job name  
#{{JOB_NAME}}
#################  
#a file for job output, you can check job progress, append the job ID with %j to make it unique
#SBATCH --output=/sailhome/clairech/robosuite/slurm_jobs/%j.out
#################
# a file for errors from the job
#SBATCH --error=/sailhome/clairech/robosuite/slurm_jobs/%j.err
#################
#time you think you need; default is 2 hours
#format could be dd-hh:mm:ss, hh:mm:ss, mm:ss, or mm
#SBATCH --time=07:00:00
#################
# Quality of Service (QOS); think of it as sending your job into a special queue; --qos=long for with a max job length of 7 days.
# uncomment ##SBATCH --qos=long if you want your job to run longer than 48 hours, which is the default for normal partition,  
# NOTE- in the hns partition the default max run time is 7 days , so you wont need to include qos, also change to normal partition 
# since dev max run time is 2 hours.
##SBATCH --qos=long
# We are submitting to the dev partition, there are several on sherlock: normal, gpu, bigmem (jobs requiring >64Gigs RAM) 
##SBATCH -p dev 
#################
# --mem is memory per node; default is 4000 MB per CPU, remember to ask for enough mem to match your CPU request, since 
# sherlock automatically allocates 4 Gigs of RAM/CPU, if you ask for 8 CPUs you will get 32 Gigs of RAM, so either 
# leave --mem commented out or request >= to the RAM needed for your CPU request.  It will also accept mem. in units, ie "--mem=4G"
#SBATCH --mem=12000
# to request multiple threads/CPUs use the -c option, on Sherlock we use 1 thread/CPU, 16 CPUs on each normal compute node 4Gigs RAM per CPU.  Here we will request just 1.
#SBATCH -c 4
#################
# Have SLURM send you an email when the job ends or fails, careful, the email could end up in your clutter folder
# Also, if you submit hundreds of jobs at once you will get hundreds of emails.
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
# Remember to change this to your email
#SBATCH --mail-user=clairech@stanford.edu
# list out some useful information
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST
echo "SLURM_NNODES="$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR
#now run normal batch commands
# Begins in slurm_scripts
BOT_MOTION="mmp"
DOOR_TYPE="dpnl"


NSTEPS=4096
HORIZON=2048
OPT_EPOCHS=10
MINIBATCHES=4
ENT_COEF=0.0
VF_COEF=0.5
CLIP_RANGE=0.2
GAMMA=0.99
LAM=0.95

RCOEF_DIST_TO_HANDLE=3
RCOEF_DOOR_ANGLE=30
RCOEF_HANDLE_CON=4
RCOEF_BODY_DOOR_CON=-5
RCOEF_SELF_CON=-2
RCOEF_ARM_HANDLE_CON=-2
RCOEF_ARM_DOOR_CON=-2
RCOEF_FORCE=-500

cd /sailhome/clairech/robosuite/robosuite/scripts

# train new learn_door.py
#python learn_door.py --job_id $SLURM_JOBID --config_file ~/robosuite/robosuite/scripts/touching_door.json --bot_motion $BOT_MOTION --door_type $DOOR_TYPE --n_steps $NSTEPS --horizon $HORIZON --opt_epochs $OPT_EPOCHS --minibatches $MINIBATCHES --ent_coef $ENT_COEF --vf_coef $VF_COEF --clip_range $CLIP_RANGE --gamma $GAMMA --lam $LAM --rcoef_dist_to_handle $RCOEF_DIST_TO_HANDLE --rcoef_door_angle $RCOEF_DOOR_ANGLE --rcoef_handle_con $RCOEF_HANDLE_CON --rcoef_body_door_con $RCOEF_BODY_DOOR_CON --rcoef_self_con $RCOEF_SELF_CON --rcoef_arm_handle_con $RCOEF_ARM_HANDLE_CON --rcoef_arm_door_con $RCOEF_ARM_DOOR_CON --rcoef_force $RCOEF_FORCE

# train from checkpoint
DIR="/sailhome/clairech/robosuite/learned_models/close_593852_a.e_bm.mmp_cr.0.2_cf.20_dp.1.3.-0.05.1.0_dq.1.0.0.-1_dt.dpnl_ec.0.0_g.0.99_h.2048_l.0.95_m.4_nc.4_ns.4096_oe.10_radc.-2.0_rahc.-2.0_rbdc.-5.0_rdth.3.0_rda.80.0_rf.-100.0_rhc.4.0_rsc.-2.0_ra.ppo2_rp.0.0.0_vc.0.5"
CF=$DIR"/args.json"
MODEL=$DIR"/5002000model.ckpt"
python learn_door.py --model $MODEL --config_file $CF


# done
echo "Done"
exit 0

