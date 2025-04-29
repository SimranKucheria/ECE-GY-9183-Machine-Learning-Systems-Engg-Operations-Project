#!/bin/bash

cd /work/deeptrust/src/continuous-x/tf/kvm
export PATH=/work/.local/bin:$PATH
unset $(set | grep -o "^OS_[A-Za-z0-9_]*")
terraform init
export TF_VAR_suffix=project3
export TF_VAR_key=MLOPS_Project3_RSA
terraform validate
terraform plan
terraform apply -auto-approve
KVM_FIP=$(terraform output -raw floating_ip_out)
cd /work/deeptrust/src/continuous-x/tf/tacc
terraform init
terraform validate
terraform plan
terraform apply -auto-approve
UC_FIP=$(terraform output -raw floating_ip_out)
cd /work/deeptrust/src/continuous-x/tf/tacc
unset $(set | grep -o "^OS_[A-Za-z0-9_]*")
terraform init
terraform validate
terraform plan
terraform apply -auto-approve
TACC_FIP=$(terraform output -raw floating_ip_out)

var1="A.B.C.D"
sed -i 's/'"$var1"'/'"$KVM_FIP"'/g' /work/deeptrust/src/continuous-x/ansible/ansible.cfg

export $KVM_IP
export $UC_IP
export $TACC_IP