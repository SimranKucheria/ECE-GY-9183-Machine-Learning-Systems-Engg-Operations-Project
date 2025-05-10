resource "openstack_compute_instance_v2" "node1" {
    name = "node1-mlops-${var.suffix}"
    image_name = "CC-Ubuntu24.04-CUDA"
    flavor_name = "baremetal"
    key_pair = var.key
    network {
      name = "sharednet1"
    }
    scheduler_hints {
      additional_properties = {
      "reservation" = "<REDACTED_RESERVATION_LEASE1>"
      }
    }
    user_data = <<-EOF
      #! /bin/bash
      sudo echo "127.0.1.1 node-mlops-${var.suffix}" >> /etc/hosts
      su cc -c /usr/local/bin/cc-load-public-keys
  EOF
}

resource "openstack_networking_floatingip_v2" "floating_ip1" {
  pool        = "public"
  description = "MLOps IP for ${var.suffix}"
}

resource "openstack_compute_floatingip_associate_v2" "fip_1" {
    floating_ip = openstack_networking_floatingip_v2.floating_ip1.address
    instance_id = openstack_compute_instance_v2.node1.id
}

resource "openstack_compute_instance_v2" "node2" {
    name = "node2-mlops-${var.suffix}"
    image_name = "CC-Ubuntu24.04-CUDA"
    flavor_name = "baremetal"
    key_pair = var.key
    network {
      name = "sharednet1"
    }
    scheduler_hints {
      additional_properties = {
      "reservation" = "<REDACTED_RESERVATION_LEASE2>"
      }
    }
    user_data = <<-EOF
      #! /bin/bash
      sudo echo "127.0.1.1 node-mlops-${var.suffix}" >> /etc/hosts
      su cc -c /usr/local/bin/cc-load-public-keys
  EOF
}

resource "openstack_networking_floatingip_v2" "floating_ip2" {
  pool        = "public"
  description = "MLOps IP for ${var.suffix}"
}

resource "openstack_compute_floatingip_associate_v2" "fip_2" {
    floating_ip = openstack_networking_floatingip_v2.floating_ip2.address
    instance_id = openstack_compute_instance_v2.node2.id
}

resource "openstack_objectstorage_container_v1" "container_1" {
  name   = "object-persist-${var.suffix}"

  metadata = {
    test = "true"
  }

  content_type = "application/json"

  versioning = true
}

