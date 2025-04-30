#!/bin/bash

export PATH=/work/.local/bin:$PATH
export PYTHONUSERBASE=/work/.local
cd /work/deeptrust/src/continuous-x/ansible
ansible-playbook -i inventory.yml pre_k8s/pre_k8s_configure.yml
cd /work/gourmetgram-iac/ansible/k8s/kubespray
ansible-playbook -i ../inventory/mycluster --become --become-user=root ./cluster.yml
cd /work/deeptrust/src/continuous-x/ansible
ansible-playbook -i inventory.yml post_k8s/post_k8s_configure.yml