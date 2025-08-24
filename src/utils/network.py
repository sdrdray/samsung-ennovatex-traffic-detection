"""
Network utilities for interface detection and network information.
"""

import socket
import platform
import subprocess
from typing import Dict, List, Optional, Any
import psutil


def get_interface_info(interface_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get information about network interfaces.
    
    Args:
        interface_name: Specific interface to get info for (None for all)
        
    Returns:
        Dictionary with interface information
    """
    try:
        interfaces = psutil.net_if_addrs()
        stats = psutil.net_if_stats()
        
        if interface_name:
            if interface_name not in interfaces:
                return {}
            
            info = {
                'name': interface_name,
                'addresses': [],
                'is_up': stats.get(interface_name, {}).get('isup', False),
                'speed': stats.get(interface_name, {}).get('speed', 0),
                'mtu': stats.get(interface_name, {}).get('mtu', 0)
            }
            
            for addr in interfaces[interface_name]:
                info['addresses'].append({
                    'family': addr.family.name,
                    'address': addr.address,
                    'netmask': addr.netmask,
                    'broadcast': addr.broadcast
                })
            
            return info
        else:
            # Return info for all interfaces
            all_info = {}
            for iface_name, addresses in interfaces.items():
                iface_stats = stats.get(iface_name, {})
                all_info[iface_name] = {
                    'name': iface_name,
                    'addresses': [
                        {
                            'family': addr.family.name,
                            'address': addr.address,
                            'netmask': addr.netmask,
                            'broadcast': addr.broadcast
                        }
                        for addr in addresses
                    ],
                    'is_up': iface_stats.get('isup', False),
                    'speed': iface_stats.get('speed', 0),
                    'mtu': iface_stats.get('mtu', 0)
                }
            
            return all_info
            
    except Exception as e:
        print(f"Error getting interface info: {e}")
        return {}


def is_interface_available(interface_name: str) -> bool:
    """
    Check if a network interface is available and up.
    
    Args:
        interface_name: Name of the interface to check
        
    Returns:
        True if interface is available and up
    """
    try:
        interfaces = psutil.net_if_addrs()
        stats = psutil.net_if_stats()
        
        if interface_name not in interfaces:
            return False
        
        # Check if interface is up
        interface_stats = stats.get(interface_name, {})
        return interface_stats.get('isup', False)
        
    except Exception:
        return False


def get_default_gateway() -> Optional[str]:
    """
    Get the default gateway IP address.
    
    Returns:
        Default gateway IP address or None if not found
    """
    try:
        gateways = psutil.net_if_stats()
        # This is a simplified approach
        # In practice, you might need platform-specific code
        
        system = platform.system().lower()
        
        if system == "windows":
            result = subprocess.run(
                ["route", "print", "0.0.0.0"],
                capture_output=True,
                text=True
            )
            # Parse Windows route output
            for line in result.stdout.split('\n'):
                if '0.0.0.0' in line and 'Gateway' not in line:
                    parts = line.split()
                    if len(parts) >= 3:
                        return parts[2]
        
        elif system == "linux":
            result = subprocess.run(
                ["ip", "route", "show", "default"],
                capture_output=True,
                text=True
            )
            # Parse Linux route output
            for line in result.stdout.split('\n'):
                if 'default via' in line:
                    parts = line.split()
                    if len(parts) >= 3:
                        return parts[2]
        
        elif system == "darwin":  # macOS
            result = subprocess.run(
                ["route", "-n", "get", "default"],
                capture_output=True,
                text=True
            )
            # Parse macOS route output
            for line in result.stdout.split('\n'):
                if 'gateway:' in line:
                    return line.split(':')[1].strip()
        
        return None
        
    except Exception:
        return None


def get_active_interfaces() -> List[str]:
    """
    Get list of active network interfaces.
    
    Returns:
        List of active interface names
    """
    try:
        interfaces = psutil.net_if_addrs()
        stats = psutil.net_if_stats()
        
        active_interfaces = []
        
        for iface_name in interfaces:
            iface_stats = stats.get(iface_name, {})
            if iface_stats.get('isup', False):
                # Check if interface has an IP address
                has_ip = any(
                    addr.family.name == 'AF_INET' 
                    for addr in interfaces[iface_name]
                )
                if has_ip:
                    active_interfaces.append(iface_name)
        
        return active_interfaces
        
    except Exception:
        return []


def test_connectivity(host: str = "8.8.8.8", port: int = 53, timeout: int = 3) -> bool:
    """
    Test network connectivity to a host.
    
    Args:
        host: Host to test connectivity to
        port: Port to test
        timeout: Connection timeout in seconds
        
    Returns:
        True if connection successful
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
        
    except Exception:
        return False


def get_network_usage() -> Dict[str, Any]:
    """
    Get current network usage statistics.
    
    Returns:
        Dictionary with network usage info
    """
    try:
        # Get network I/O statistics
        net_io = psutil.net_io_counters()
        
        # Get per-interface statistics
        net_io_per_nic = psutil.net_io_counters(pernic=True)
        
        return {
            'total': {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv,
                'errin': net_io.errin,
                'errout': net_io.errout,
                'dropin': net_io.dropin,
                'dropout': net_io.dropout
            },
            'per_interface': {
                iface: {
                    'bytes_sent': stats.bytes_sent,
                    'bytes_recv': stats.bytes_recv,
                    'packets_sent': stats.packets_sent,
                    'packets_recv': stats.packets_recv,
                    'errin': stats.errin,
                    'errout': stats.errout,
                    'dropin': stats.dropin,
                    'dropout': stats.dropout
                }
                for iface, stats in net_io_per_nic.items()
            }
        }
        
    except Exception as e:
        return {'error': str(e)}


def is_port_available(port: int, host: str = 'localhost') -> bool:
    """
    Check if a port is available for binding.
    
    Args:
        port: Port number to check
        host: Host to check on
        
    Returns:
        True if port is available
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((host, port))
        sock.close()
        return True
        
    except socket.error:
        return False


def get_local_ip() -> Optional[str]:
    """
    Get the local IP address of this machine.
    
    Returns:
        Local IP address or None if not found
    """
    try:
        # Connect to a remote server to determine local IP
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        local_ip = sock.getsockname()[0]
        sock.close()
        return local_ip
        
    except Exception:
        try:
            # Fallback method
            hostname = socket.gethostname()
            return socket.gethostbyname(hostname)
        except Exception:
            return None
