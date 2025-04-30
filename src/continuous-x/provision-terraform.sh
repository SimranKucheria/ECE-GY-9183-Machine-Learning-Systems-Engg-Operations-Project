#!/bin/bash

export PATH=/work/.local/bin:$PATH
export TF_VAR_suffix=project3
export TF_VAR_key=MLOPS_Project3_RSA
unset $(set | grep -o "^OS_[A-Za-z0-9_]*")
cd /work/deeptrust/src/continuous-x/tf/kvm
terraform init
terraform validate
terraform plan
terraform apply -auto-approve
KVM_FIP=$(terraform output -raw floating_ip_out)
cd /work/deeptrust/src/continuous-x/tf/uc
terraform init
terraform validate
terraform plan
terraform apply -auto-approve
UC_FIP=$(terraform output -raw floating_ip_out)
cd /work/deeptrust/src/continuous-x/tf/tacc
terraform init
terraform validate
terraform plan
terraform apply -auto-approve
TACC_FIP=$(terraform output -raw floating_ip_out)

var1="A.B.C.D"
sed -i 's/'"$var1"'/'"$KVM_FIP"'/g' /work/deeptrust/src/continuous-x/ansible.cfg

echo export KVM_FIP="${KVM_FIP}" >> ~/.bashrc
echo export TACC_FIP="${TACC_FIP}" >> ~/.bashrc
echo export UC_FIP="${UC_FIP}" >> ~/.bashrc

cp /work/deeptrust/src/continuous-x/ansible.cfg /work/deeptrust/src/continuous-x/ansible/ansible.cfg
     