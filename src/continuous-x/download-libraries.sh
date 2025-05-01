#!/bin/bash

mkdir -p /work/.local/bin
wget https://releases.hashicorp.com/terraform/1.10.5/terraform_1.10.5_linux_amd64.zip
unzip -o -q terraform_1.10.5_linux_amd64.zip
mv terraform /work/.local/bin
rm terraform_1.10.5_linux_amd64.zip

cp clouds.yaml /work/deeptrust/src/continuous-x/tf/kvm/clouds.yaml
cp clouds.yaml /work/deeptrust/src/continuous-x/tf/tacc/clouds.yaml
cp clouds.yaml /work/deeptrust/src/continuous-x/tf/uc/clouds.yaml

export PATH=/work/.local/bin:$PATH

PYTHONUSERBASE=/work/.local pip install --user ansible-core==2.16.9 ansible==9.8.0

# export PATH=/work/.local/bin:$PATH
# export PYTHONUSERBASE=/work/.local

# PYTHONUSERBASE=/work/.local pip install --user -r /work/deeptrust/src/continuous-x/ansible/k8s/kubespray/requirements.txt


