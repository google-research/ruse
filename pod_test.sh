PROJECT=ruse-xgcp
DISKTYPE=pd-standard
SERVICEACCT=726688283486-compute@developer.gserviceaccount.com

: <<'EOF'
gcloud compute instance-templates create podtest \
    --machine-type n1-standard-16 \
    --image-project=ml-images \
    --image=debian-9-torch-xla-v20201203 \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --network=projects/$PROJECT/global/networks/default \
    --network-tier=PREMIUM --maintenance-policy=MIGRATE \
    --service-account=$SERVICEACCT --scopes=https://www.googleapis.com/auth/cloud-platform,cloud-platform,storage-full,cloud-source-repos \
    --boot-disk-size=2000GB --boot-disk-type=$DISKTYPE --reservation-affinity=any --tags=deeplearning-vm,ssh-tunnel-iap


gcloud compute instance-groups managed create podgroup \
    --size 4 \
    --template podtest \
    --zone europe-west4-a

gcloud compute instance-groups list-instances podgroup
gcloud compute ssh  podgroup-50vp --zone=europe-west4-a --tunnel-through-iap\

gcloud compute tpus create tpu-pod \
    --zone=europe-west4-a \
    --network=default \
    --accelerator-type=v2-32 \
    --version=pytorch-1.7
EOF

# inside the machine we ssh to 
export TPU_NAME=tpu-pod
conda activate torch-xla-1.7

python -m torch_xla.distributed.xla_dist \
      --tpu=$TPU_NAME \
      --conda-env=torch-xla-1.7 \
      --env XLA_USE_BF16=1 \
      -- python /usr/share/torch-xla-1.7/pytorch/xla/test/test_train_mp_imagenet.py \
      --fake_data
