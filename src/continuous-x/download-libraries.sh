#!/bin/bash
git clone --recurse-submodules https://github.com/SimranKucheria/ECE-GY-9183-Machine-Learning-Systems-Engg-Operations-Project.git /work/deeptrust

mkdir -p /work/.local/bin
wget https://releases.hashicorp.com/terraform/1.10.5/terraform_1.10.5_linux_amd64.zip
unzip -o -q terraform_1.10.5_linux_amd64.zip
mv terraform /work/.local/bin
rm terraform_1.10.5_linux_amd64.zip

cp clouds.yaml /work/deeptrust/tf/kvm/clouds.yaml
cp clouds.yaml /work/deeptrust/tf/tacc/clouds.yaml
cp clouds.yaml /work/deeptrust/tf/uc/clouds.yaml

export PATH=/work/.local/bin:$PATH

PYTHONUSERBASE=/work/.local pip install --user ansible-core==2.16.9 ansible==9.8.0

export PATH=/work/.local/bin:$PATH
export PYTHONUSERBASE=/work/.local

PYTHONUSERBASE=/work/.local pip install --user -r /work/deeptrust/ansible/k8s/kubespray/requirements.txt


