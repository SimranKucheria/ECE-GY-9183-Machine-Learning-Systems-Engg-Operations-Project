resource "openstack_networking_port_v2" "sharednet1_ports" {

    name       = "sharednet1-mlops-${var.suffix}"
    network_id = data.openstack_networking_network_v2.sharednet1.id
}


resource "openstack_compute_instance_v2" "node" {
    name = "node-mlops-${var.suffix}"
    image_name = "CC-Ubuntu20.04"
    flavor_name = "baremetal"
    key_pair = var.key
    network {
      name = openstack_networking_port_v2.sharednet1_ports.id
    }
    scheduler_hints {
      additional_properties = {
      "reservation" = "<REDACTED_RESERVATION_LEASE>"
      }
    }
    user_data = <<-EOF
      #! /bin/bash
      sudo echo "127.0.1.1 node-mlops-${var.suffix}" >> /etc/hosts
      su cc -c /usr/local/bin/cc-load-public-keys
  EOF
}

resource "openstack_networking_floatingip_v2" "floating_ip" {
  pool        = "public"
  description = "MLOps IP for ${var.suffix}"
  port_id     = openstack_networking_port_v2.sharednet1_ports.id
}

