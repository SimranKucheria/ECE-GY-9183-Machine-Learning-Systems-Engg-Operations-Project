output "floating_ip_out1" {
  description = "Floating IP assigned to node1"
  value       = openstack_networking_floatingip_v2.floating_ip1.address
}

output "floating_ip_out2" {
  description = "Floating IP assigned to node2"
  value       = openstack_networking_floatingip_v2.floating_ip2.address
}

