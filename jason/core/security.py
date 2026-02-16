"""
J.A.S.O.N. Aegis Protocol
Iron Shield Advanced: Active Firewall Management, Network Sniffing & Offensive Security
"""

import os
import subprocess
import threading
import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import re

# Network sniffing dependencies
try:
    import scapy.all as scapy
    scapy_available = True
except ImportError:
    scapy_available = False

logger = logging.getLogger(__name__)

class AegisManager:
    """Aegis Protocol: Iron Shield Advanced with Offensive Capabilities"""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.sniffing_active = False
        self.sniff_thread: Optional[threading.Thread] = None
        self.captured_packets = []
        self.max_packets = 1000

        # Firewall settings
        self.firewall_rules = []
        self.blocked_ips = set()
        self.allowed_ports = set()

        # Offensive security settings
        self.pentest_targets = []
        self.scan_results = {}
        self.vulnerability_db = self._load_vulnerability_database()

        # Alert thresholds
        self.suspicious_packet_threshold = 50  # packets per minute
        self.alert_interval = 60  # seconds

        # Initialize firewall
        self._initialize_firewall()

    def _load_vulnerability_database(self) -> Dict[str, Any]:
        """Load vulnerability database"""
        vuln_file = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'vulnerabilities.json')
        try:
            if os.path.exists(vuln_file):
                with open(vuln_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load vulnerability database: {e}")
        return {}

    def _save_vulnerability_database(self):
        """Save vulnerability database"""
        vuln_file = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'vulnerabilities.json')
        os.makedirs(os.path.dirname(vuln_file), exist_ok=True)
        try:
            with open(vuln_file, 'w') as f:
                json.dump(self.vulnerability_db, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save vulnerability database: {e}")

    def _initialize_firewall(self):
        """Initialize firewall with basic rules"""
        try:
            # Check if PF is enabled
            result = subprocess.run(['pfctl', '-s', 'info'],
                                  capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                logger.info("Packet Filter (PF) is active")
                self._load_current_rules()
            else:
                logger.warning("Packet Filter (PF) may not be active")
                # Try to enable PF
                self._enable_pf()

        except Exception as e:
            logger.error(f"Failed to initialize firewall: {e}")

    def _enable_pf(self):
        """Enable Packet Filter on macOS"""
        try:
            # Enable PF
            subprocess.run(['sudo', 'pfctl', '-e'], check=True)
            logger.info("Packet Filter enabled")

            # Load basic rules
            self._create_basic_pf_rules()

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to enable PF: {e}")

    def _create_basic_pf_rules(self):
        """Create basic PF rules file"""
        rules = """# J.A.S.O.N. Aegis Protocol Basic Rules
block all
pass in on lo0 all
pass out all keep state
"""

        rules_file = "/etc/pf.jason.conf"
        try:
            # Write rules to file
            with open(rules_file, 'w') as f:
                f.write(rules)

            # Load rules
            subprocess.run(['sudo', 'pfctl', '-f', rules_file], check=True)
            logger.info("Basic PF rules loaded")

        except Exception as e:
            logger.error(f"Failed to create PF rules: {e}")

    def _load_current_rules(self):
        """Load current firewall rules"""
        try:
            result = subprocess.run(['pfctl', '-s', 'rules'],
                                  capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                self.firewall_rules = result.stdout.strip().split('\n')
                logger.info(f"Loaded {len(self.firewall_rules)} firewall rules")

        except Exception as e:
            logger.error(f"Failed to load firewall rules: {e}")

    def add_block_rule(self, ip_address: str) -> bool:
        """Add a block rule for an IP address"""
        try:
            # Add to blocked IPs set
            self.blocked_ips.add(ip_address)

            # Add to PF
            subprocess.run(['sudo', 'pfctl', '-t', 'jason_blocked', '-T', 'add', ip_address],
                         check=True)

            # Reload rules if needed
            self._reload_firewall()

            logger.info(f"Blocked IP: {ip_address}")
            return True

        except Exception as e:
            logger.error(f"Failed to block IP {ip_address}: {e}")
            return False

    def remove_block_rule(self, ip_address: str) -> bool:
        """Remove a block rule for an IP address"""
        try:
            if ip_address in self.blocked_ips:
                self.blocked_ips.remove(ip_address)

            # Remove from PF table
            subprocess.run(['sudo', 'pfctl', '-t', 'jason_blocked', '-T', 'delete', ip_address],
                         check=True)

            logger.info(f"Unblocked IP: {ip_address}")
            return True

        except Exception as e:
            logger.error(f"Failed to unblock IP {ip_address}: {e}")
            return False

    def allow_port(self, port: int, protocol: str = 'tcp') -> bool:
        """Allow traffic on a specific port"""
        try:
            self.allowed_ports.add((port, protocol))

            # Add rule to PF
            rule = f"pass in proto {protocol} to port {port}"

            # For now, just log - full implementation would modify pf.conf
            logger.info(f"Allowed {protocol} port {port}")

            # Reload rules
            self._reload_firewall()
            return True

        except Exception as e:
            logger.error(f"Failed to allow port {port}: {e}")
            return False

    def block_port(self, port: int, protocol: str = 'tcp') -> bool:
        """Block traffic on a specific port"""
        try:
            self.allowed_ports.discard((port, protocol))

            # Add block rule
            rule = f"block drop proto {protocol} to port {port}"

            logger.info(f"Blocked {protocol} port {port}")

            # Reload rules
            self._reload_firewall()
            return True

        except Exception as e:
            logger.error(f"Failed to block port {port}: {e}")
            return False

    def _reload_firewall(self):
        """Reload firewall rules"""
        try:
            subprocess.run(['sudo', 'pfctl', '-f', '/etc/pf.jason.conf'],
                         check=True, timeout=10)
            logger.info("Firewall rules reloaded")

        except Exception as e:
            logger.error(f"Failed to reload firewall: {e}")

    def start_network_sniffing(self, interface: str = 'en0', promiscuous: bool = True) -> bool:
        """Start network packet sniffing"""
        if not scapy_available:
            logger.error("Scapy not available for network sniffing")
            return False

        if self.sniffing_active:
            logger.warning("Network sniffing already active")
            return False

        try:
            self.sniffing_active = True
            self.sniff_thread = threading.Thread(
                target=self._sniff_packets,
                args=(interface, promiscuous),
                daemon=True
            )
            self.sniff_thread.start()

            logger.info(f"Started network sniffing on interface {interface}")
            return True

        except Exception as e:
            logger.error(f"Failed to start network sniffing: {e}")
            self.sniffing_active = False
            return False

    def stop_network_sniffing(self) -> bool:
        """Stop network packet sniffing"""
        if not self.sniffing_active:
            return False

        try:
            self.sniffing_active = False
            if self.sniff_thread:
                self.sniff_thread.join(timeout=5)

            logger.info("Stopped network sniffing")
            return True

        except Exception as e:
            logger.error(f"Failed to stop network sniffing: {e}")
            return False

    def _sniff_packets(self, interface: str, promiscuous: bool):
        """Packet sniffing loop"""
        def packet_callback(packet):
            if not self.sniffing_active:
                return

            # Analyze packet
            packet_info = self._analyze_packet(packet)

            # Store packet info
            self.captured_packets.append(packet_info)

            # Limit packet storage
            if len(self.captured_packets) > self.max_packets:
                self.captured_packets.pop(0)

            # Check for suspicious activity
            self._check_suspicious_activity(packet_info)

        try:
            # Start sniffing
            scapy.sniff(
                iface=interface,
                prn=packet_callback,
                store=0,  # Don't store packets in memory
                promisc=promiscuous
            )

        except Exception as e:
            logger.error(f"Packet sniffing error: {e}")
            self.sniffing_active = False

    def _analyze_packet(self, packet) -> Dict[str, Any]:
        """Analyze a captured packet"""
        info = {
            'timestamp': datetime.now().isoformat(),
            'length': len(packet),
            'summary': str(packet.summary())
        }

        # Extract IP information
        if packet.haslayer('IP'):
            ip_layer = packet['IP']
            info.update({
                'src_ip': ip_layer.src,
                'dst_ip': ip_layer.dst,
                'protocol': ip_layer.proto,
                'ttl': ip_layer.ttl
            })

        # Extract TCP/UDP information
        if packet.haslayer('TCP'):
            tcp_layer = packet['TCP']
            info.update({
                'src_port': tcp_layer.sport,
                'dst_port': tcp_layer.dport,
                'tcp_flags': tcp_layer.flags
            })
        elif packet.haslayer('UDP'):
            udp_layer = packet['UDP']
            info.update({
                'src_port': udp_layer.sport,
                'dst_port': udp_layer.dport
            })

        return info

    def _check_suspicious_activity(self, packet_info: Dict[str, Any]):
        """Check packet for suspicious activity"""
        src_ip = packet_info.get('src_ip')
        dst_port = packet_info.get('dst_ip')

        # Check for blocked IPs
        if src_ip and src_ip in self.blocked_ips:
            logger.warning(f"Suspicious packet from blocked IP: {src_ip}")
            return

        # Check for port scanning (multiple connections to different ports)
        # This is a simplified check
        if dst_port and dst_port not in [80, 443, 22, 53]:  # Common ports
            # Count packets to this port in last minute
            recent_packets = [p for p in self.captured_packets[-100:]
                            if p.get('dst_ip') == packet_info.get('dst_ip')
                            and p.get('dst_port') == dst_port]

            if len(recent_packets) > self.suspicious_packet_threshold:
                logger.warning(f"Potential port scan detected on port {dst_port}")
                # Could auto-block here

    def get_firewall_status(self) -> Dict[str, Any]:
        """Get current firewall status"""
        return {
            'pf_enabled': self._check_pf_status(),
            'blocked_ips': list(self.blocked_ips),
            'allowed_ports': list(self.allowed_ports),
            'total_rules': len(self.firewall_rules)
        }

    def get_sniffing_status(self) -> Dict[str, Any]:
        """Get network sniffing status"""
        return {
            'active': self.sniffing_active,
            'captured_packets': len(self.captured_packets),
            'max_packets': self.max_packets
        }

    def get_recent_packets(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent captured packets"""
        return self.captured_packets[-limit:] if self.captured_packets else []

    def _check_pf_status(self) -> bool:
        """Check if Packet Filter is enabled"""
        try:
            result = subprocess.run(['pfctl', '-s', 'info'],
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except:
            return False

    def export_firewall_config(self, filepath: str) -> bool:
        """Export current firewall configuration"""
        try:
            config = {
                'blocked_ips': list(self.blocked_ips),
                'allowed_ports': list(self.allowed_ports),
                'firewall_rules': self.firewall_rules,
                'exported_at': datetime.now().isoformat()
            }

            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)

            logger.info(f"Firewall config exported to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to export firewall config: {e}")
            return False

    def import_firewall_config(self, filepath: str) -> bool:
        """Import firewall configuration"""
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)

            # Restore blocked IPs
            self.blocked_ips = set(config.get('blocked_ips', []))

            # Restore allowed ports
            self.allowed_ports = set(tuple(p) for p in config.get('allowed_ports', []))

            # Reload rules
            self._reload_firewall()

            logger.info(f"Firewall config imported from {filepath}")
            return False

        except Exception as e:
            logger.error(f"Failed to import firewall config: {e}")
            return False

    # ========== OFFENSIVE SECURITY FEATURES ==========

    def network_scan(self, target: str, scan_type: str = 'basic') -> Dict[str, Any]:
        """Perform network scanning using Nmap"""
        try:
            logger.info(f"Starting {scan_type} scan on {target}")

            if scan_type == 'basic':
                # Basic host discovery and port scan
                cmd = ['nmap', '-sV', '-O', '--version-light', target]
            elif scan_type == 'comprehensive':
                # Comprehensive scan
                cmd = ['nmap', '-sS', '-sV', '-O', '-A', '--version-all', target]
            elif scan_type == 'vulnerability':
                # Vulnerability scan
                cmd = ['nmap', '--script', 'vulners', target]
            elif scan_type == 'stealth':
                # Stealth scan
                cmd = ['nmap', '-sS', '-T2', target]
            else:
                return {'success': False, 'error': f'Unknown scan type: {scan_type}'}

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            scan_result = {
                'success': result.returncode == 0,
                'target': target,
                'scan_type': scan_type,
                'output': result.stdout if result.returncode == 0 else result.stderr,
                'timestamp': datetime.now().isoformat()
            }

            if scan_result['success']:
                # Parse and store vulnerabilities
                self._parse_scan_results(target, scan_result['output'])
                logger.info(f"Scan completed successfully for {target}")
            else:
                logger.error(f"Scan failed for {target}: {scan_result['output']}")

            return scan_result

        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Scan timed out', 'target': target}
        except Exception as e:
            logger.error(f"Scan error for {target}: {e}")
            return {'success': False, 'error': str(e), 'target': target}

    def _parse_scan_results(self, target: str, output: str):
        """Parse Nmap scan results for vulnerabilities"""
        vulnerabilities = []

        # Extract version information and potential vulnerabilities
        lines = output.split('\n')
        current_service = None

        for line in lines:
            if 'open' in line and '/' in line:
                # Parse service line
                parts = line.split()
                if len(parts) >= 3:
                    port = parts[0]
                    state = parts[1]
                    service = parts[2]
                    current_service = f"{port}/{service}"

            elif current_service and ('vulnerable' in line.lower() or 'exploit' in line.lower()):
                # Found potential vulnerability
                vulnerabilities.append({
                    'service': current_service,
                    'description': line.strip(),
                    'severity': 'high' if 'critical' in line.lower() else 'medium'
                })

        if vulnerabilities:
            self.vulnerability_db[target] = {
                'scan_date': datetime.now().isoformat(),
                'vulnerabilities': vulnerabilities
            }
            self._save_vulnerability_database()

    def exploit_vulnerability(self, target: str, vulnerability: str) -> Dict[str, Any]:
        """Attempt to exploit a vulnerability (AUTHORIZED PENTESTING ONLY)"""
        logger.warning(f"EXPLOIT ATTEMPT: {vulnerability} on {target}")
        logger.warning("ENSURE YOU HAVE WRITTEN AUTHORIZATION FOR THIS PENTEST")

        # This is a framework - actual exploitation would require specific modules
        # and should only be used with explicit authorization

        return {
            'success': False,
            'message': 'Exploit framework initialized - manual authorization required',
            'target': target,
            'vulnerability': vulnerability,
            'timestamp': datetime.now().isoformat()
        }

    def honey_pot_detection(self, interface: str = 'en0') -> Dict[str, Any]:
        """Detect potential honey pots and deceptive network devices"""
        suspicious_devices = []

        try:
            # Use ARP scanning to find devices
            result = subprocess.run(['arp', '-a'], capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if '(' in line and ')' in line:
                        # Parse ARP entry
                        parts = line.split()
                        if len(parts) >= 2:
                            ip = parts[1].strip('()')
                            mac = parts[3] if len(parts) > 3 else 'unknown'

                            # Check for suspicious patterns
                            if self._is_suspicious_device(ip, mac):
                                suspicious_devices.append({
                                    'ip': ip,
                                    'mac': mac,
                                    'reason': 'Potential honey pot or monitoring device'
                                })

        except Exception as e:
            logger.error(f"Honey pot detection error: {e}")

        return {
            'devices_scanned': len(result.stdout.split('\n')) if 'result' in locals() else 0,
            'suspicious_devices': suspicious_devices,
            'timestamp': datetime.now().isoformat()
        }

    def _is_suspicious_device(self, ip: str, mac: str) -> bool:
        """Check if a device appears suspicious"""
        # Check for known virtualization MAC prefixes
        virtual_mac_prefixes = ['08:00:27', '00:0C:29', '00:50:56', '52:54:00']

        for prefix in virtual_mac_prefixes:
            if mac.startswith(prefix):
                return True

        # Check for unusual IP patterns
        if ip.startswith('192.168.') or ip.startswith('10.') or ip.startswith('172.'):
            # Common private ranges - check for anomalies
            octets = ip.split('.')
            if len(octets) == 4:
                # Suspicious if all octets are the same or sequential
                if len(set(octets)) == 1 or all(int(octets[i]) == int(octets[0]) + i for i in range(4)):
                    return True

        return False

    def man_in_the_middle_attack(self, target_ip: str, gateway_ip: str) -> Dict[str, Any]:
        """Perform ARP poisoning for MITM (AUTHORIZED PENTESTING ONLY)"""
        logger.warning(f"MITM ATTEMPT: ARP poisoning between {target_ip} and {gateway_ip}")
        logger.warning("ENSURE YOU HAVE WRITTEN AUTHORIZATION FOR THIS PENTEST")

        if not scapy_available:
            return {'success': False, 'error': 'Scapy not available for MITM'}

        try:
            # Get MAC addresses
            target_mac = self._get_mac(target_ip)
            gateway_mac = self._get_mac(gateway_ip)

            if not target_mac or not gateway_mac:
                return {'success': False, 'error': 'Could not resolve MAC addresses'}

            # Send ARP poison packets (this is for demonstration only)
            # In real scenarios, this would be more sophisticated
            logger.warning("ARP POISONING DEMONSTRATION - NOT EXECUTED")
            logger.warning("This is a simulation. Actual MITM requires careful setup and authorization.")

            return {
                'success': True,
                'message': 'MITM framework ready - execution requires authorization',
                'target_ip': target_ip,
                'target_mac': target_mac,
                'gateway_ip': gateway_ip,
                'gateway_mac': gateway_mac,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _get_mac(self, ip: str) -> Optional[str]:
        """Get MAC address for IP using ARP"""
        try:
            if scapy_available:
                ans, _ = scapy.arping(ip, verbose=0, timeout=2)
                for snd, rcv in ans:
                    return rcv[scapy.ARP].hwsrc
        except:
            pass
        return None

    def get_vulnerability_report(self, target: str = None) -> Dict[str, Any]:
        """Generate vulnerability report"""
        if target:
            if target in self.vulnerability_db:
                return self.vulnerability_db[target]
            else:
                return {'error': f'No vulnerability data for {target}'}
        else:
            return self.vulnerability_db

    def initiate_pentest(self, target: str, scope: str = 'basic') -> Dict[str, Any]:
        """Initiate authorized penetration testing sequence"""
        logger.warning(f"PENTEST INITIATION: {scope} test on {target}")
        logger.warning("CONFIRM AUTHORIZATION BEFORE PROCEEDING")

        pentest_plan = {
            'target': target,
            'scope': scope,
            'timestamp': datetime.now().isoformat(),
            'phases': []
        }

        if scope == 'basic':
            pentest_plan['phases'] = [
                'reconnaissance',
                'scanning',
                'enumeration',
                'reporting'
            ]
        elif scope == 'comprehensive':
            pentest_plan['phases'] = [
                'reconnaissance',
                'scanning',
                'enumeration',
                'vulnerability_assessment',
                'exploitation',
                'post_exploitation',
                'reporting'
            ]

        # Add to pentest targets
        self.pentest_targets.append(pentest_plan)

        return {
            'success': True,
            'message': f'Pentest plan created for {target}',
            'plan': pentest_plan
        }

    def get_aegis_status(self) -> Dict[str, Any]:
        """Get comprehensive Aegis Protocol status"""
        return {
            'firewall_status': self.get_firewall_status(),
            'sniffing_status': self.get_sniffing_status(),
            'vulnerability_targets': list(self.vulnerability_db.keys()),
            'active_pentests': len(self.pentest_targets),
            'total_vulnerabilities_found': sum(len(v.get('vulnerabilities', [])) for v in self.vulnerability_db.values())
        }
    """Iron Shield Protocol: Active firewall management and network sniffing"""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.sniffing_active = False
        self.sniff_thread: Optional[threading.Thread] = None
        self.captured_packets = []
        self.max_packets = 1000

        # Firewall settings
        self.firewall_rules = []
        self.blocked_ips = set()
        self.allowed_ports = set()

        # Alert thresholds
        self.suspicious_packet_threshold = 50  # packets per minute
        self.alert_interval = 60  # seconds

        # Initialize firewall
        self._initialize_firewall()

    def _initialize_firewall(self):
        """Initialize firewall with basic rules"""
        try:
            # Check if PF is enabled
            result = subprocess.run(['pfctl', '-s', 'info'],
                                  capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                logger.info("Packet Filter (PF) is active")
                self._load_current_rules()
            else:
                logger.warning("Packet Filter (PF) may not be active")
                # Try to enable PF
                self._enable_pf()

        except Exception as e:
            logger.error(f"Failed to initialize firewall: {e}")

    def _enable_pf(self):
        """Enable Packet Filter on macOS"""
        try:
            # Enable PF
            subprocess.run(['sudo', 'pfctl', '-e'], check=True)
            logger.info("Packet Filter enabled")

            # Load basic rules
            self._create_basic_pf_rules()

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to enable PF: {e}")

    def _create_basic_pf_rules(self):
        """Create basic PF rules file"""
        rules = """# J.A.S.O.N. Iron Shield Basic Rules
block all
pass in on lo0 all
pass out all keep state
"""

        rules_file = "/etc/pf.jason.conf"
        try:
            # Write rules to file
            with open(rules_file, 'w') as f:
                f.write(rules)

            # Load rules
            subprocess.run(['sudo', 'pfctl', '-f', rules_file], check=True)
            logger.info("Basic PF rules loaded")

        except Exception as e:
            logger.error(f"Failed to create PF rules: {e}")

    def _load_current_rules(self):
        """Load current firewall rules"""
        try:
            result = subprocess.run(['pfctl', '-s', 'rules'],
                                  capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                self.firewall_rules = result.stdout.strip().split('\n')
                logger.info(f"Loaded {len(self.firewall_rules)} firewall rules")

        except Exception as e:
            logger.error(f"Failed to load firewall rules: {e}")

    def add_block_rule(self, ip_address: str) -> bool:
        """Add a block rule for an IP address"""
        try:
            # Add to blocked IPs set
            self.blocked_ips.add(ip_address)

            # Create rule
            rule = f"block drop from {ip_address} to any"

            # Add to PF
            subprocess.run(['sudo', 'pfctl', '-t', 'jason_blocked', '-T', 'add', ip_address],
                         check=True)

            # Reload rules if needed
            self._reload_firewall()

            logger.info(f"Blocked IP: {ip_address}")
            return True

        except Exception as e:
            logger.error(f"Failed to block IP {ip_address}: {e}")
            return False

    def remove_block_rule(self, ip_address: str) -> bool:
        """Remove a block rule for an IP address"""
        try:
            if ip_address in self.blocked_ips:
                self.blocked_ips.remove(ip_address)

            # Remove from PF table
            subprocess.run(['sudo', 'pfctl', '-t', 'jason_blocked', '-T', 'delete', ip_address],
                         check=True)

            logger.info(f"Unblocked IP: {ip_address}")
            return True

        except Exception as e:
            logger.error(f"Failed to unblock IP {ip_address}: {e}")
            return False

    def allow_port(self, port: int, protocol: str = 'tcp') -> bool:
        """Allow traffic on a specific port"""
        try:
            self.allowed_ports.add((port, protocol))

            # Add rule to PF
            rule = f"pass in proto {protocol} to port {port}"

            # For now, just log - full implementation would modify pf.conf
            logger.info(f"Allowed {protocol} port {port}")

            # Reload rules
            self._reload_firewall()
            return True

        except Exception as e:
            logger.error(f"Failed to allow port {port}: {e}")
            return False

    def block_port(self, port: int, protocol: str = 'tcp') -> bool:
        """Block traffic on a specific port"""
        try:
            self.allowed_ports.discard((port, protocol))

            # Add block rule
            rule = f"block drop proto {protocol} to port {port}"

            logger.info(f"Blocked {protocol} port {port}")

            # Reload rules
            self._reload_firewall()
            return True

        except Exception as e:
            logger.error(f"Failed to block port {port}: {e}")
            return False

    def _reload_firewall(self):
        """Reload firewall rules"""
        try:
            subprocess.run(['sudo', 'pfctl', '-f', '/etc/pf.jason.conf'],
                         check=True, timeout=10)
            logger.info("Firewall rules reloaded")

        except Exception as e:
            logger.error(f"Failed to reload firewall: {e}")

    def start_network_sniffing(self, interface: str = 'en0', promiscuous: bool = True) -> bool:
        """Start network packet sniffing"""
        if not scapy_available:
            logger.error("Scapy not available for network sniffing")
            return False

        if self.sniffing_active:
            logger.warning("Network sniffing already active")
            return False

        try:
            self.sniffing_active = True
            self.sniff_thread = threading.Thread(
                target=self._sniff_packets,
                args=(interface, promiscuous),
                daemon=True
            )
            self.sniff_thread.start()

            logger.info(f"Started network sniffing on interface {interface}")
            return True

        except Exception as e:
            logger.error(f"Failed to start network sniffing: {e}")
            self.sniffing_active = False
            return False

    def stop_network_sniffing(self) -> bool:
        """Stop network packet sniffing"""
        if not self.sniffing_active:
            return False

        try:
            self.sniffing_active = False
            if self.sniff_thread:
                self.sniff_thread.join(timeout=5)

            logger.info("Stopped network sniffing")
            return True

        except Exception as e:
            logger.error(f"Failed to stop network sniffing: {e}")
            return False

    def _sniff_packets(self, interface: str, promiscuous: bool):
        """Packet sniffing loop"""
        def packet_callback(packet):
            if not self.sniffing_active:
                return

            # Analyze packet
            packet_info = self._analyze_packet(packet)

            # Store packet info
            self.captured_packets.append(packet_info)

            # Limit packet storage
            if len(self.captured_packets) > self.max_packets:
                self.captured_packets.pop(0)

            # Check for suspicious activity
            self._check_suspicious_activity(packet_info)

        try:
            # Start sniffing
            scapy.sniff(
                iface=interface,
                prn=packet_callback,
                store=0,  # Don't store packets in memory
                promisc=promiscuous
            )

        except Exception as e:
            logger.error(f"Packet sniffing error: {e}")
            self.sniffing_active = False

    def _analyze_packet(self, packet) -> Dict[str, Any]:
        """Analyze a captured packet"""
        info = {
            'timestamp': datetime.now().isoformat(),
            'length': len(packet),
            'summary': str(packet.summary())
        }

        # Extract IP information
        if packet.haslayer('IP'):
            ip_layer = packet['IP']
            info.update({
                'src_ip': ip_layer.src,
                'dst_ip': ip_layer.dst,
                'protocol': ip_layer.proto,
                'ttl': ip_layer.ttl
            })

        # Extract TCP/UDP information
        if packet.haslayer('TCP'):
            tcp_layer = packet['TCP']
            info.update({
                'src_port': tcp_layer.sport,
                'dst_port': tcp_layer.dport,
                'tcp_flags': tcp_layer.flags
            })
        elif packet.haslayer('UDP'):
            udp_layer = packet['UDP']
            info.update({
                'src_port': udp_layer.sport,
                'dst_port': udp_layer.dport
            })

        return info

    def _check_suspicious_activity(self, packet_info: Dict[str, Any]):
        """Check packet for suspicious activity"""
        src_ip = packet_info.get('src_ip')
        dst_port = packet_info.get('dst_port')

        # Check for blocked IPs
        if src_ip and src_ip in self.blocked_ips:
            logger.warning(f"Suspicious packet from blocked IP: {src_ip}")
            return

        # Check for port scanning (multiple connections to different ports)
        # This is a simplified check
        if dst_port and dst_port not in [80, 443, 22, 53]:  # Common ports
            # Count packets to this port in last minute
            recent_packets = [p for p in self.captured_packets[-100:]
                            if p.get('dst_ip') == packet_info.get('dst_ip')
                            and p.get('dst_port') == dst_port]

            if len(recent_packets) > self.suspicious_packet_threshold:
                logger.warning(f"Potential port scan detected on port {dst_port}")
                # Could auto-block here

    def get_firewall_status(self) -> Dict[str, Any]:
        """Get current firewall status"""
        return {
            'pf_enabled': self._check_pf_status(),
            'blocked_ips': list(self.blocked_ips),
            'allowed_ports': list(self.allowed_ports),
            'total_rules': len(self.firewall_rules)
        }

    def get_sniffing_status(self) -> Dict[str, Any]:
        """Get network sniffing status"""
        return {
            'active': self.sniffing_active,
            'captured_packets': len(self.captured_packets),
            'max_packets': self.max_packets
        }

    def get_recent_packets(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent captured packets"""
        return self.captured_packets[-limit:] if self.captured_packets else []

    def _check_pf_status(self) -> bool:
        """Check if Packet Filter is enabled"""
        try:
            result = subprocess.run(['pfctl', '-s', 'info'],
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except:
            return False

    def export_firewall_config(self, filepath: str) -> bool:
        """Export current firewall configuration"""
        try:
            config = {
                'blocked_ips': list(self.blocked_ips),
                'allowed_ports': list(self.allowed_ports),
                'firewall_rules': self.firewall_rules,
                'exported_at': datetime.now().isoformat()
            }

            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)

            logger.info(f"Firewall config exported to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to export firewall config: {e}")
            return False

    def import_firewall_config(self, filepath: str) -> bool:
        """Import firewall configuration"""
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)

            # Restore blocked IPs
            self.blocked_ips = set(config.get('blocked_ips', []))

            # Restore allowed ports
            self.allowed_ports = set(tuple(p) for p in config.get('allowed_ports', []))

            # Reload rules
            self._reload_firewall()

            logger.info(f"Firewall config imported from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to import firewall config: {e}")
            return False
