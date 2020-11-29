#!/bin/bash

# bash gcp_launch.sh 8 20201128 fixed-attentitive-scheduled path_to_config
################### PARAMS BEGIN
VERSION=1.7 #nightly #1.7
NCORES=$1
if [ -z $2 ]
then
  #DATE=`date -d "1 day ago" +%Y%m%d`
  DATE=`date +%Y%m%d`
  echo "no date specified, using $DATE, and version $VERSION"
else
  DATE=$2
  echo "using $DATE, and version $VERSION"
fi
DISKIMG=debian-9-torch-xla-v$DATE #internship #debian-9-torch-xla-v$DATE
if [ -z $1 ]
then
  echo "give tpu cores as first arg"
  exit
fi
TASKNAME=$3
config=$4
PROJECT=ruse-xgcp
MTYPE=n1-highmem-96 #e2-standard-32
SERVICEACCT=726688283486-compute@developer.gserviceaccount.com
ZONE=europe-west4-a
DISKTYPE=pd-standard
code_dir=/usr/local/google/home/rabeeh/Desktop/internship
output_dir=/usr/local/google/home/rabeeh/Desktop/
#DISKTYPE=pd-ssd
################### PARAMS END
# ml-images
set -e
set -x
# Create instance template
IT=$TASKNAME-it #internship #$TASKNAME-it
# XXX: create resource
# --image-project=ruse-xgcp
gcloud beta compute --project=$PROJECT instance-templates create $IT --machine-type=$MTYPE --network=projects/$PROJECT/global/networks/default --network-tier=PREMIUM --maintenance-policy=MIGRATE --service-account=$SERVICEACCT --scopes=https://www.googleapis.com/auth/cloud-platform,cloud-platform,storage-full,cloud-source-repos --image=$DISKIMG --image-project=ml-images --boot-disk-size=2000GB --boot-disk-type=$DISKTYPE --boot-disk-device-name=$IT --reservation-affinity=any --tags=deeplearning-vm,ssh-tunnel-iap

# create instance group
IG=$TASKNAME-ig
igsize=$(( NCORES / 8 ))

# XXX: create resource
gcloud compute --project=$PROJECT instance-groups managed create $IG --base-instance-name=$IG --template=$IT --size=$igsize --zone=$ZONE

# create TPU
TPU=$TASKNAME-tpu-$NCORES
# XXX: create resource
gcloud compute tpus create $TPU --project=$PROJECT --zone=$ZONE --network=default --version=pytorch-$VERSION --accelerator-type=v3-$NCORES
echo $TPU

TPU_IP_ADDRESS=$(gcloud compute tpus describe --zone=$ZONE $TPU --format "value(ipAddress)")
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
echo $XRT_TPU_CONFIG


#cp -r internship $remotepath
#pip install -r $remotepath/internship/seq2seq/requirements.txt
# prepare vms
gcloud compute instance-groups managed wait-until --stable $IG --zone $ZONE
CONDAENV=torch-xla-$VERSION
pids=
for instance in $(gcloud --project=$PROJECT compute instance-groups managed list-instances $IG --zone=$ZONE --format='value(NAME)[terminator=" "]')
do
  #localpath=$code_dir/requirements.txt
  remotepath=/home/rabeeh/
  #gcloud compute scp --project=$PROJECT --zone=$ZONE $localpath/requirements.txt $instance:$remotepath
  gcloud compute scp --recurse --project=$PROJECT --zone=$ZONE $code_dir $instance:$remotepath  --tunnel-through-iap
  gcloud compute ssh --project=$PROJECT --zone=$ZONE "$instance" --tunnel-through-iap --command "
    
    . /anaconda3/etc/profile.d/conda.sh
    conda activate $CONDAENV
    cd $remotepath/internship/
    python setup.py develop
    pip install -r $remotepath/internship/requirements.txt
    # TODO: prepare the vms, download data, install code, whatever.
" &
  pids+=" $!"
done
wait $pids || { echo "failed initializing vms" >&2; exit 1; }

# Wait for healthy tpu
TPU_HEALTHY_ATTEMPTS=300
set +x
echo 'Waiting for tpu to become healthy.'
for x in `seq 1 $TPU_HEALTHY_ATTEMPTS`
do
  echo "  Polling, $x"
  _TPU_HEALTHY=`gcloud compute tpus describe $TPU --zone=$ZONE`
  _health_substr="health: HEALTHY"
  if [[ $_TPU_HEALTHY == *"$_health_substr"* ]]; then
    echo "After $x tries, tpu is healthy."
    break
  fi
  sleep 5
done
set -x
test $TPU_HEALTHY_ATTEMPTS -gt $x
if [ -z $2 ]
then
  echo "No date specified, using nightly tpu runtime"
else
  echo "Updating TPU runtime to $DATE"
  gcloud compute ssh --project=$PROJECT --zone=$ZONE "$instance" --tunnel-through-iap --command "
    set -e  # Fail on first failure
    set -x  # echo commands
    GVM_IPS=\$(gcloud --project=${PROJECT} compute tpus describe \
      ${TPU} --zone=$ZONE --format='value(networkEndpoints[].ipAddress)')
    pids=
    for ip in \$(echo \$GVM_IPS | tr ';' '\n')
    do
      curl -X POST http://\${ip}:8475/requestversion/pytorch-dev$DATE &
      pids+=\" \$!\"
    done
    wait \$pids
  "
fi
TPU_HEALTHY_ATTEMPTS=300
set +x
echo 'Waiting for tpu to become healthy.'
for x in `seq 1 $TPU_HEALTHY_ATTEMPTS`
do
  echo "  Polling, $x"
  _TPU_HEALTHY=`gcloud compute tpus describe $TPU --zone=$ZONE`
  _health_substr="health: HEALTHY"
  if [[ $_TPU_HEALTHY == *"$_health_substr"* ]]; then
    echo "After $x tries, tpu is healthy."
    break
  fi
  sleep 5
done

# RUN THE TRAINING JOB:
set +e
set -x
gcloud compute ssh --project=$PROJECT --zone=$ZONE $instance --tunnel-through-iap --command "
  . /anaconda3/etc/profile.d/conda.sh
  conda activate $CONDAENV
  export XRT_TPU_CONFIG=\"$XRT_TPU_CONFIG\"
  echo \"\$XRT_TPU_CONFIG\"
  cd $remotepath/internship/seq2seq/
  python $remotepath/internship/seq2seq/xla_spawn.py \
  $remotepath/internship/seq2seq/finetune_t5_trainer.py \
  $remotepath/internship/$config
"

#fixed_length_emb/attentive.json
gcloud compute scp --recurse --project=$PROJECT --zone=$ZONE $instance:$remotepath/internship/seq2seq/outputs $output_dir  --tunnel-through-iap

# python -m torch_xla.distributed.xla_dist \
#   --tpu=$TPU \
#   --conda-env=$CONDAENV \
#   -- python3 $remotepath/internship/seq2seq/finetune_t5_trainer.py $remotepath/internship/seq2seq/configs/test_xla.json | tee logfile.txt
#gcloud compute scp --project=$PROJECT --zone=$ZONE $instance:$pathtomodel $localpath/model.pt

# CLEANUP
/usr/bin/yes | gcloud compute instance-groups managed delete $IG --zone=$ZONE
/usr/bin/yes | gcloud compute tpus delete $TPU --project=$PROJECT --zone=$ZONE
/usr/bin/yes | gcloud compute instance-templates delete $IT
