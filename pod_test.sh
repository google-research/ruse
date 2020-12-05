gcloud compute instance-templates create instance-template-name \
    --machine-type n1-standard-16 \
    --image-project=ruse-xgcp \
    --image=gcr.io/tpu-pytorch/xla:r1.7 \
    --scopes=https://www.googleapis.com/auth/cloud-platform
