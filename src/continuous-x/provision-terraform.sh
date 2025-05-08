#!/bin/bash

export PATH=/work/.local/bin:$PATH
export TF_VAR_suffix=project3
export TF_VAR_key=id_rsa_chameleon
unset $(set | grep -o "^OS_[A-Za-z0-9_]*")
cd /work/deeptrust/src/continuous-x/tf/kvm
terraform init
terraform validate
terraform plan
terraform apply -auto-approve
KVM_FIP=$(terraform output -raw floating_ip_out)
cd /work/deeptrust/src/continuous-x/tf/tacc
terraform init
terraform validate
terraform plan
terraform apply -auto-approve
TACC_FIP1=$(terraform output -raw floating_ip_out1)
TACC_FIP2=$(terraform output -raw floating_ip_out2)

var1="A.B.C.D"
sed -i 's/'"$var1"'/'"$KVM_FIP"'/g' /work/deeptrust/src/continuous-x/ansible.cfg

sed -i 's/'"$var1"'/'$TACC_FIP1'/g' /work/deeptrust/src/continuous-x/ansible/ansible-baremetal/inventory.yaml

var2="E.F.G.H"
sed -i 's/'"$var2"'/'$TACC_FIP2'/g' /work/deeptrust/src/continuous-x/ansible/ansible-baremetal/inventory.yaml

echo export KVM_FIP="${KVM_FIP}" >> ~/.bashrc
echo export TACC_FIP1="${TACC_FIP1}" >> ~/.bashrc
echo export TACC_FIP2="${TACC_FIP2}" >> ~/.bashrc

cp /work/deeptrust/src/continuous-x/ansible.cfg /work/deeptrust/src/continuous-x/ansible/ansible.cfg
     