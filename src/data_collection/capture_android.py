#!/usr/bin/env python3
"""
Android Traffic Capture Module

Captures network traffic from Android devices using multiple methods:
1. USB tethering with PC-based capture
2. WiFi hotspot monitoring
3. VPN service integration (requires custom Android app)
"""

import os
import sys
import time
import logging
import argparse
import subprocess
import threading
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path

import psutil

from ..utils.logging import setup_logger
from ..utils.network import get_interface_info, is_interface_available, test_connectivity


class AndroidCapture:
    """
    Android network traffic capture using various methods.
    
    Supports USB tethering, WiFi hotspot, and VPN service approaches
    for capturing mobile app traffic data.
    """
    
    def __init__(self, method: str = "tethering", output_dir: str = "data/raw"):
        """
        Initialize Android capture instance.
        
        Args:
            method: Capture method ('tethering', 'hotspot', 'vpn')
            output_dir: Directory to store capture files
        """
        self.method = method.lower()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logger(f"android_capture_{self.method}")
        self.is_capturing = False
        self.capture_process = None
        
        # Method-specific configuration
        self.config = self._get_method_config()
        
        # Validate method and setup
        self._validate_setup()
    
    def _get_method_config(self) -> Dict[str, Any]:
        """Get configuration for the selected capture method."""
        configs = {
            'tethering': {
                'interface_patterns': ['usb', 'rndis', 'tether'],
                'adb_required': True,
                'description': 'USB tethering with ADB bridge'
            },
            'hotspot': {
                'interface_patterns': ['wlan', 'wifi', 'ap'],
                'adb_required': False,
                'description': 'WiFi hotspot monitoring'
            },
            'vpn': {
                'interface_patterns': ['tun', 'tap'],
                'adb_required': True,
                'description': 'VPN service integration'
            }
        }
        
        if self.method not in configs:
            raise ValueError(f"Unsupported method: {self.method}")
        
        return configs[self.method]
    
    def _validate_setup(self):
        """Validate that the system is set up for the selected method."""
        self.logger.info(f"Validating setup for method: {self.config['description']}")
        
        # Check for ADB if required
        if self.config['adb_required']:
            if not self._check_adb():
                raise RuntimeError("ADB not found or not accessible")
        
        # Check platform-specific requirements
        if sys.platform == "win32":
            self._validate_windows_setup()
        elif sys.platform.startswith("linux"):
            self._validate_linux_setup()
        elif sys.platform == "darwin":
            self._validate_macos_setup()
    
    def _check_adb(self) -> bool:
        """Check if ADB is available and device is connected."""
        try:
            result = subprocess.run(
                ["adb", "devices"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                self.logger.error("ADB command failed")
                return False
            
            # Parse ADB output
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            connected_devices = [line for line in lines if 'device' in line and 'offline' not in line]
            
            if not connected_devices:
                self.logger.error("No Android devices connected via ADB")
                return False
            
            self.logger.info(f"Found {len(connected_devices)} connected Android device(s)")
            return True
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.logger.error("ADB not found or timeout")
            return False
    
    def _validate_windows_setup(self):
        """Validate Windows-specific setup."""
        if self.method == "tethering":
            # Check for USB tethering interface
            interfaces = get_interface_info()
            tethering_interfaces = [
                name for name in interfaces
                if any(pattern in name.lower() for pattern in self.config['interface_patterns'])
            ]
            
            if not tethering_interfaces:
                self.logger.warning("No USB tethering interface found. Make sure device is connected and tethering is enabled.")
    
    def _validate_linux_setup(self):
        """Validate Linux-specific setup."""
        # Check if user has necessary permissions
        if os.geteuid() != 0:
            self.logger.warning("Running without root privileges. Some capture methods may not work.")
        
        # Check for required tools
        tools = ['tcpdump']
        if self.method == "tethering":
            tools.append('iptables')
        
        for tool in tools:
            if not self._command_exists(tool):
                raise RuntimeError(f"Required tool '{tool}' not found")
    
    def _validate_macos_setup(self):
        """Validate macOS-specific setup."""
        # Similar to Linux but with macOS-specific considerations
        if self.method == "tethering":
            # Check for USB tethering support
            pass
    
    def _command_exists(self, command: str) -> bool:
        """Check if a command exists on the system."""
        try:
            subprocess.run(
                ["which", command],
                capture_output=True,
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def setup_tethering(self) -> bool:
        """
        Set up USB tethering capture.
        
        Returns:
            True if setup successful
        """
        try:
            self.logger.info("Setting up USB tethering...")
            
            # Enable USB tethering via ADB
            result = subprocess.run(
                ["adb", "shell", "svc", "usb", "setFunctions", "rndis"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                self.logger.error(f"Failed to enable tethering: {result.stderr}")
                return False
            
            # Wait for interface to come up
            time.sleep(3)
            
            # Find tethering interface
            tethering_interface = self._find_tethering_interface()
            if not tethering_interface:
                self.logger.error("Tethering interface not found")
                return False
            
            self.tethering_interface = tethering_interface
            self.logger.info(f"Tethering interface: {tethering_interface}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Tethering setup failed: {e}")
            return False
    
    def _find_tethering_interface(self) -> Optional[str]:
        """Find the USB tethering network interface."""
        interfaces = get_interface_info()
        
        for name, info in interfaces.items():
            if any(pattern in name.lower() for pattern in self.config['interface_patterns']):
                if info['is_up']:
                    return name
        
        return None
    
    def setup_hotspot_monitoring(self) -> bool:
        """
        Set up WiFi hotspot monitoring.
        
        Returns:
            True if setup successful
        """
        try:
            self.logger.info("Setting up WiFi hotspot monitoring...")
            
            # This requires the phone to be set up as a WiFi hotspot
            # and the PC to connect to it for monitoring
            
            # Find WiFi interface
            wifi_interface = self._find_wifi_interface()
            if not wifi_interface:
                self.logger.error("WiFi interface not found")
                return False
            
            self.wifi_interface = wifi_interface
            self.logger.info(f"WiFi interface: {wifi_interface}")
            
            # Test connectivity to ensure we're connected to phone's hotspot
            if not test_connectivity():
                self.logger.warning("No internet connectivity detected")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Hotspot setup failed: {e}")
            return False
    
    def _find_wifi_interface(self) -> Optional[str]:
        """Find the WiFi network interface."""
        interfaces = get_interface_info()
        
        for name, info in interfaces.items():
            if any(pattern in name.lower() for pattern in ['wlan', 'wifi', 'wireless']):
                if info['is_up']:
                    return name
        
        return None
    
    def start_capture(self, duration: Optional[int] = None, apps: List[str] = None) -> str:
        """
        Start Android traffic capture.
        
        Args:
            duration: Capture duration in seconds
            apps: List of specific apps to capture (optional)
            
        Returns:
            Path to output file
        """
        if self.is_capturing:
            raise RuntimeError("Capture already in progress")
        
        # Setup method-specific capture
        if self.method == "tethering":
            if not hasattr(self, 'tethering_interface'):
                if not self.setup_tethering():
                    raise RuntimeError("Tethering setup failed")
            interface = self.tethering_interface
        elif self.method == "hotspot":
            if not hasattr(self, 'wifi_interface'):
                if not self.setup_hotspot_monitoring():
                    raise RuntimeError("Hotspot setup failed")
            interface = self.wifi_interface
        else:
            raise RuntimeError(f"Method {self.method} not implemented")
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        apps_str = "_".join(apps) if apps else "all"
        output_file = self.output_dir / f"android_{self.method}_{apps_str}_{timestamp}.pcap"
        
        self.logger.info(f"Starting Android capture via {self.method}")
        self.logger.info(f"Interface: {interface}")
        self.logger.info(f"Output file: {output_file}")
        
        try:
            # Start capture process
            self.is_capturing = True
            self._start_capture_process(interface, output_file, duration, apps)
            
            return str(output_file)
            
        except Exception as e:
            self.is_capturing = False
            self.logger.error(f"Error starting capture: {e}")
            raise
    
    def _start_capture_process(self, interface: str, output_file: Path, 
                             duration: Optional[int], apps: List[str]):
        """Start the actual capture process."""
        
        # Build tcpdump command
        cmd = ["tcpdump", "-i", interface, "-w", str(output_file)]
        
        # Add filters for social media traffic
        packet_filter = self._build_packet_filter(apps)
        if packet_filter:
            cmd.append(packet_filter)
        
        # Add duration if specified
        if duration:
            cmd.extend(["-G", str(duration), "-W", "1"])
        
        self.logger.info(f"Running command: {' '.join(cmd)}")
        
        try:
            # Start capture process
            self.capture_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Monitor process
            self._monitor_capture_process(duration)
            
        except Exception as e:
            self.logger.error(f"Capture process failed: {e}")
            raise
    
    def _build_packet_filter(self, apps: List[str]) -> str:
        """
        Build packet filter for specific apps.
        
        Args:
            apps: List of app names
            
        Returns:
            BPF filter string
        """
        if not apps:
            # Default filter for HTTPS traffic
            return "tcp port 443 or tcp port 80"
        
        # App-specific domain filters
        domain_filters = []
        for app in apps:
            if app.lower() == 'instagram':
                domain_filters.extend(['instagram.com', 'cdninstagram.com'])
            elif app.lower() == 'tiktok':
                domain_filters.extend(['tiktok.com', 'tiktokcdn.com'])
            elif app.lower() == 'youtube':
                domain_filters.extend(['youtube.com', 'googlevideo.com'])
            # Add more apps as needed
        
        if domain_filters:
            host_filter = " or ".join([f"host {domain}" for domain in domain_filters])
            return f"(tcp port 443 or tcp port 80) and ({host_filter})"
        else:
            return "tcp port 443 or tcp port 80"
    
    def _monitor_capture_process(self, duration: Optional[int]):
        """Monitor the capture process."""
        try:
            if duration:
                # Wait for specified duration
                self.capture_process.wait(timeout=duration + 10)
            else:
                # Wait indefinitely
                self.capture_process.wait()
                
        except subprocess.TimeoutExpired:
            self.logger.info("Capture duration completed")
            self.stop_capture()
        except Exception as e:
            self.logger.error(f"Error monitoring capture: {e}")
        finally:
            self.is_capturing = False
    
    def stop_capture(self):
        """Stop ongoing capture."""
        if not self.is_capturing:
            self.logger.warning("No capture in progress")
            return
        
        self.logger.info("Stopping Android capture...")
        
        if self.capture_process:
            try:
                self.capture_process.terminate()
                self.capture_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.capture_process.kill()
                self.capture_process.wait()
            
            self.capture_process = None
        
        self.is_capturing = False
        self.logger.info("Capture stopped")
    
    def get_connected_devices(self) -> List[Dict[str, str]]:
        """
        Get list of connected Android devices.
        
        Returns:
            List of device information dictionaries
        """
        try:
            result = subprocess.run(
                ["adb", "devices", "-l"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                return []
            
            devices = []
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            
            for line in lines:
                if 'device' in line and 'offline' not in line:
                    parts = line.split()
                    device_id = parts[0]
                    
                    # Get device info
                    info_result = subprocess.run(
                        ["adb", "-s", device_id, "shell", "getprop", "ro.product.model"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    
                    model = info_result.stdout.strip() if info_result.returncode == 0 else "Unknown"
                    
                    devices.append({
                        'id': device_id,
                        'model': model,
                        'status': 'connected'
                    })
            
            return devices
            
        except Exception as e:
            self.logger.error(f"Error getting devices: {e}")
            return []


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Android Network Traffic Capture")
    parser.add_argument('--method', '-m', choices=['tethering', 'hotspot', 'vpn'], 
                       default='tethering', help="Capture method")
    parser.add_argument('--duration', '-d', type=int, help="Capture duration in seconds")
    parser.add_argument('--apps', '-a', nargs='+', help="Specific apps to capture")
    parser.add_argument('--output', '-o', default="data/raw", help="Output directory")
    parser.add_argument('--verbose', '-v', action='store_true', help="Verbose logging")
    parser.add_argument('--list-devices', action='store_true', help="List connected devices")
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Initialize capture
        capture = AndroidCapture(method=args.method, output_dir=args.output)
        
        # List devices if requested
        if args.list_devices:
            devices = capture.get_connected_devices()
            if devices:
                print("Connected Android devices:")
                for device in devices:
                    print(f"  {device['id']} - {device['model']} ({device['status']})")
            else:
                print("No Android devices found")
            return
        
        # Start capture
        output_file = capture.start_capture(duration=args.duration, apps=args.apps)
        print(f"Android capture started. Output: {output_file}")
        
        # Wait for completion or user interrupt
        try:
            if args.duration:
                print(f"Capturing for {args.duration} seconds...")
                time.sleep(args.duration)
            else:
                print("Press Ctrl+C to stop capture...")
                while capture.is_capturing:
                    time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping capture...")
        
        # Stop capture
        capture.stop_capture()
        print("Android capture completed")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
