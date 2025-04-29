resource "openstack_networking_port_v2" "sharednet1_ports" {
  for_each   = var.nodes
    name       = "sharednet1-${each.key}-mlops-${var.suffix}"
    network_id = data.openstack_networking_network_v2.sharednet1.id
}


resource "openstack_compute_instance_v2" "nodes" {
    for_each = var.nodes

    name = "${each.key}-mlops-${var.suffix}"
    image_name = "CC-Ubuntu20.04"
    flavor_name = "baremetal"
    key_pair = var.key
    network {
      name = openstack_networking_port_v2.sharednet1_ports[each.key].id
    }
    scheduler_hints {
      additional_properties = {
      "reservation" = "90d25f80-f25f-44b5-8168-d0767c5c0bba"
      }
    }
    user_data = <<-EOF
      #! /bin/bash
      sudo echo "127.0.1.1 ${each.key}-mlops-${var.suffix}" >> /etc/hosts
      su cc -c /usr/local/bin/cc-load-public-keys
  EOF
}

resource "openstack_networking_floatingip_v2" "floating_ip" {
  pool        = "public"
  description = "MLOps IP for ${var.suffix}"
  port_id     = openstack_networking_port_v2.sharednet1_ports["node1"].id
}

