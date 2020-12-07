#!/bin/bash

################### PARAMS BEGIN
VERSION=nightly  # 1.7
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
DISKIMG=debian-9-torch-xla-v$DATE
if [ -z $1 ]
then
  echo "give tpu cores as first arg"
  exit
fi
TASKNAME=rabeeh-t5-pods
PROJECT= #TODO: fill
MTYPE=e2-standard-32 # TODO: fill
SERVICEACCT=  # TODO: fill
ZONE=europe-west4-a
DISKTYPE=pd-standard
#DISKTYPE=pd-ssd
################### PARAMS END

set -e
set -x
# Create instance template
IT=$TASKNAME-it
# XXX: create resource
gcloud beta compute --project=$PROJECT instance-templates create $IT --machine-type=$MTYPE --network=projects/$PROJECT/global/networks/default --network-tier=PREMIUM --maintenance-policy=MIGRATE --service-account=$SERVICEACCT --scopes=https://www.googleapis.com/auth/cloud-platform --image=$DISKIMG --image-project=ml-images --boot-disk-size=200GB --boot-disk-type=$DISKTYPE --boot-disk-device-name=$IT --reservation-affinity=any

# create instance group
IG=$TASKNAME-ig
igsize=$(( NCORES / 8 ))

# XXX: create resource
gcloud compute --project=$PROJECT instance-groups managed create $IG --base-instance-name=$IG --template=$IT --size=$igsize --zone=$ZONE

# create TPU
TPU=$TASKNAME-tpu-$NCORES
# XXX: create resource
gcloud compute tpus create $TPU --project=$PROJECT --zone=$ZONE --network=default --version=pytorch-$VERSION --accelerator-type=v3-$NCORES

# prepare vms
gcloud compute instance-groups managed wait-until --stable $IG --zone $ZONE
CONDAENV=torch-xla-$VERSION
pids=
for instance in $(gcloud --project=$PROJECT compute instance-groups managed list-instances $IG --zone=$ZONE --format='value(NAME)[terminator=" "]')
do
  localpath=/path/to/requirements.txt
  remotepath=/home/rabeeh/github/blablabla/
  gcloud compute scp --project=$PROJECT --zone=$ZONE $localpath/requirements.txt $instance:$remotepath
  gcloud compute ssh --project=$PROJECT --zone=$ZONE "$instance" --command "
    . /anaconda3/etc/profile.d/conda.sh
    conda activate $CONDAENV
    pip install -r $remotepath/requirements.txt
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
  gcloud compute ssh --project=$PROJECT --zone=$ZONE "$instance" --command "
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
gcloud compute ssh --project=$PROJECT --zone=$ZONE $instance --command "
  . /anaconda3/etc/profile.d/conda.sh
  conda activate $CONDAENV
  python -m torch_xla.distributed.xla_dist \
    --tpu=$TPU \
    --conda-env=$CONDAENV \
    -- python3 # TODO: rabeeh's command | tee logs/logfile.txt
"

gcloud compute scp --project=$PROJECT --zone=$ZONE $instance:$pathtomodel $localpath/model.pt

# CLEANUP
#/usr/bin/yes | gcloud compute instance-groups managed delete $IG --zone=$ZONE
/usr/bin/yes | gcloud compute tpus delete $TPU --project=$PROJECT --zone=$ZONE
/usr/bin/yes | gcloud compute instance-templates delete $IT
