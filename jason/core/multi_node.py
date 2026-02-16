"""
J.A.S.O.N. Multi-Node Ghost Control - Cross-Device Consciousness
Neural Handoff & Omnipresent Sync: Seamless State Transfer Across Ecosystem
"""

import requests
import json
import threading
import time
from typing import Dict, List, Any, Optional, Set
from langchain_core.tools import Tool
import socket
import uuid
import hashlib
from datetime import datetime, timedelta
import queue

class GhostControlManager:
    """Ghost Control: Neural handoff across AR glasses, smartwatches, PC, and home servers"""

    def __init__(self):
        # Device registry with enhanced metadata
        self.devices = {}  # device_id: device_info
        self.device_states = {}  # device_id: current_state
        self.state_transfers = {}  # transfer_id: transfer_info

        # Consciousness stream - shared across all devices
        self.consciousness_stream = queue.Queue(maxsize=1000)
        self.shared_context = {
            'active_user': None,
            'current_task': None,
            'environment_state': {},
            'user_intent': {},
            'device_hierarchy': {},
            'last_sync': None
        }

        # Device hierarchy for intelligent routing
        self.device_hierarchy = {
            "master": "pc",
            "primaries": ["ar_glasses", "smartwatch"],
            "secondaries": ["phone", "tablet", "laptop"],
            "tertiaries": ["smart_home_hub", "security_camera", "thermostat"],
            "capabilities": {
                "pc": ["compute", "vision", "audio", "storage", "network", "display"],
                "ar_glasses": ["display", "vision", "audio", "location", "biometrics"],
                "smartwatch": ["notification", "health", "location", "audio", "haptic"],
                "phone": ["mobile", "camera", "location", "audio", "display"],
                "tablet": ["display", "compute", "touch", "camera"],
                "laptop": ["compute", "display", "network", "storage"],
                "smart_home_hub": ["iot", "automation", "sensors", "network"],
                "security_camera": ["vision", "motion", "recording"],
                "thermostat": ["climate", "sensors", "automation"]
            },
            "power_levels": {
                "pc": "unlimited",
                "ar_glasses": "low",
                "smartwatch": "very_low",
                "phone": "medium",
                "tablet": "medium",
                "laptop": "high",
                "smart_home_hub": "low",
                "security_camera": "low",
                "thermostat": "very_low"
            },
            "context_priorities": {
                "ar_glasses": 1,  # Highest priority for user-facing
                "smartwatch": 2,
                "pc": 3,
                "phone": 4,
                "laptop": 5,
                "tablet": 6,
                "smart_home_hub": 7,
                "security_camera": 8,
                "thermostat": 9
            }
        }

        # Neural handoff settings
        self.handoff_active = False
        self.handoff_thread = None
        self.neural_bridge_active = False
        self.bridge_thread = None

        # State synchronization
        self.sync_interval = 1.0  # seconds
        self.last_sync = time.time()
        self.sync_queue = queue.Queue()

        # Device discovery and registration
        self.discovery_active = False
        self.discovery_thread = None
        self.known_devices = set()

        # Initialize core devices
        self._initialize_core_devices()

        # Create tools
        self.ghost_handoff_tool = Tool(
            name="Ghost Handoff",
            description="Initiate neural handoff to another device in the ecosystem",
            func=self.initiate_ghost_handoff
        )

        self.consciousness_sync_tool = Tool(
            name="Consciousness Sync",
            description="Synchronize consciousness state across all active devices",
            func=self.synchronize_consciousness
        )

        self.device_discovery_tool = Tool(
            name="Device Discovery",
            description="Discover and register new devices in the Ghost ecosystem",
            func=self.discover_devices
        )

        self.state_transfer_tool = Tool(
            name="State Transfer",
            description="Transfer current state to another device seamlessly",
            func=self.transfer_state
        )

    def _initialize_core_devices(self):
        """Initialize core devices in the Ghost ecosystem"""
        core_devices = [
            {
                'id': 'pc',
                'name': 'Primary PC',
                'type': 'computer',
                'capabilities': self.device_hierarchy['capabilities']['pc'],
                'power_level': 'unlimited',
                'status': 'active',
                'last_seen': datetime.now().isoformat(),
                'ip_address': self._get_local_ip(),
                'consciousness_level': 'full'
            },
            {
                'id': 'ar_glasses',
                'name': 'AR Glasses',
                'type': 'wearable',
                'capabilities': self.device_hierarchy['capabilities']['ar_glasses'],
                'power_level': 'low',
                'status': 'inactive',
                'last_seen': None,
                'ip_address': None,
                'consciousness_level': 'minimal'
            },
            {
                'id': 'smartwatch',
                'name': 'Smartwatch',
                'type': 'wearable',
                'capabilities': self.device_hierarchy['capabilities']['smartwatch'],
                'power_level': 'very_low',
                'status': 'inactive',
                'last_seen': None,
                'ip_address': None,
                'consciousness_level': 'minimal'
            }
        ]

        for device in core_devices:
            self.devices[device['id']] = device
            self.device_states[device['id']] = {
                'consciousness_active': device['id'] == 'pc',
                'context_loaded': device['id'] == 'pc',
                'last_state': {},
                'pending_transfers': []
            }

    def _get_local_ip(self) -> str:
        """Get local IP address"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"

    def initiate_ghost_handoff(self, target_device: str, handoff_type: str = "seamless") -> Dict[str, Any]:
        """Initiate neural handoff to another device"""
        if target_device not in self.devices:
            return {'success': False, 'error': f'Device {target_device} not found in ecosystem'}

        if not self.devices[target_device].get('status') == 'active':
            return {'success': False, 'error': f'Device {target_device} is not active'}

        try:
            # Create handoff transfer
            transfer_id = str(uuid.uuid4())
            current_device = self._get_current_primary_device()

            transfer_data = {
                'transfer_id': transfer_id,
                'source_device': current_device,
                'target_device': target_device,
                'handoff_type': handoff_type,
                'timestamp': datetime.now().isoformat(),
                'state_snapshot': self._create_state_snapshot(current_device),
                'consciousness_context': self.shared_context.copy(),
                'status': 'initiating'
            }

            self.state_transfers[transfer_id] = transfer_data

            # Start handoff process
            if handoff_type == "seamless":
                success = self._execute_seamless_handoff(transfer_data)
            elif handoff_type == "context_only":
                success = self._execute_context_handoff(transfer_data)
            else:
                return {'success': False, 'error': f'Unknown handoff type: {handoff_type}'}

            if success:
                # Update device states
                self.device_states[current_device]['consciousness_active'] = False
                self.device_states[target_device]['consciousness_active'] = True

                transfer_data['status'] = 'completed'
                self._broadcast_handoff_completion(transfer_data)

                return {
                    'success': True,
                    'transfer_id': transfer_id,
                    'message': f'Ghost handoff completed: {current_device} → {target_device}'
                }
            else:
                transfer_data['status'] = 'failed'
                return {'success': False, 'error': 'Handoff execution failed'}

        except Exception as e:
            return {'success': False, 'error': f'Handoff failed: {str(e)}'}

    def _execute_seamless_handoff(self, transfer_data: Dict[str, Any]) -> bool:
        """Execute seamless neural handoff"""
        try:
            source = transfer_data['source_device']
            target = transfer_data['target_device']

            # Phase 1: Prepare target device
            self._prepare_target_device(target, transfer_data)

            # Phase 2: Transfer consciousness stream
            self._transfer_consciousness_stream(source, target)

            # Phase 3: Synchronize context
            self._synchronize_context(source, target)

            # Phase 4: Complete handoff
            self._finalize_handoff(source, target)

            return True

        except Exception as e:
            print(f"Seamless handoff failed: {e}")
            return False

    def _execute_context_handoff(self, transfer_data: Dict[str, Any]) -> bool:
        """Execute context-only handoff (lighter weight)"""
        try:
            source = transfer_data['source_device']
            target = transfer_data['target_device']

            # Transfer only essential context
            context_data = {
                'user_intent': self.shared_context.get('user_intent', {}),
                'current_task': self.shared_context.get('current_task'),
                'environment_state': self.shared_context.get('environment_state', {})
            }

            self._send_context_to_device(target, context_data)

            # Update consciousness pointer
            self.shared_context['active_device'] = target

            return True

        except Exception as e:
            print(f"Context handoff failed: {e}")
            return False

    def _create_state_snapshot(self, device_id: str) -> Dict[str, Any]:
        """Create a snapshot of device state for transfer"""
        return {
            'device_id': device_id,
            'timestamp': datetime.now().isoformat(),
            'consciousness_state': self.device_states[device_id],
            'active_processes': [],  # Would include actual process state
            'memory_state': {},  # Would include memory context
            'ui_state': {}  # Would include UI/interface state
        }

    def _prepare_target_device(self, device_id: str, transfer_data: Dict[str, Any]):
        """Prepare target device for consciousness transfer"""
        # In a real implementation, this would:
        # 1. Wake device if asleep
        # 2. Prepare memory allocation
        # 3. Initialize consciousness receptors
        # 4. Establish neural bridge connection

        print(f"Preparing {device_id} for consciousness transfer...")

    def _transfer_consciousness_stream(self, source_id: str, target_id: str):
        """Transfer the active consciousness stream"""
        # In implementation, this would transfer:
        # - Active thought processes
        # - Current task state
        # - User interaction context
        # - Environmental awareness

        print(f"Transferring consciousness stream: {source_id} → {target_id}")

    def _synchronize_context(self, source_id: str, target_id: str):
        """Synchronize shared context between devices"""
        context_data = self.shared_context.copy()
        self._send_context_to_device(target_id, context_data)

    def _finalize_handoff(self, source_id: str, target_id: str):
        """Finalize the handoff process"""
        # Update shared context
        self.shared_context['active_device'] = target_id
        self.shared_context['last_handoff'] = datetime.now().isoformat()

        # Send completion signal to both devices
        self._send_completion_signal(source_id, target_id)

    def _send_context_to_device(self, device_id: str, context: Dict[str, Any]):
        """Send context data to a specific device"""
        # In implementation, this would use device-specific protocols
        # (Bluetooth, WiFi, cellular, etc.)
        device = self.devices.get(device_id)
        if device and device.get('ip_address'):
            try:
                # Send via HTTP API
                url = f"http://{device['ip_address']}:5001/context"
                requests.post(url, json=context, timeout=5)
            except:
                # Device not reachable, store for later sync
                self._queue_context_for_later(device_id, context)

    def _broadcast_handoff_completion(self, transfer_data: Dict[str, Any]):
        """Broadcast handoff completion to all devices"""
        completion_msg = {
            'type': 'handoff_completed',
            'transfer_data': transfer_data,
            'new_active_device': transfer_data['target_device']
        }

        for device_id, device in self.devices.items():
            if device.get('status') == 'active':
                self._send_message_to_device(device_id, completion_msg)

    def _get_current_primary_device(self) -> str:
        """Get the currently active primary device"""
        for device_id, state in self.device_states.items():
            if state.get('consciousness_active', False):
                return device_id
        return 'pc'  # Default fallback

    def synchronize_consciousness(self, scope: str = "all") -> Dict[str, Any]:
        """Synchronize consciousness state across devices"""
        try:
            sync_data = {
                'timestamp': datetime.now().isoformat(),
                'shared_context': self.shared_context,
                'active_device': self._get_current_primary_device(),
                'device_states': self.device_states.copy()
            }

            synced_devices = 0

            if scope == "all":
                target_devices = [d for d in self.devices.keys() if self.devices[d].get('status') == 'active']
            elif scope in self.devices:
                target_devices = [scope]
            else:
                return {'success': False, 'error': f'Invalid sync scope: {scope}'}

            for device_id in target_devices:
                if self._send_sync_to_device(device_id, sync_data):
                    synced_devices += 1

            return {
                'success': True,
                'devices_synced': synced_devices,
                'total_targeted': len(target_devices),
                'sync_timestamp': sync_data['timestamp']
            }

        except Exception as e:
            return {'success': False, 'error': f'Synchronization failed: {str(e)}'}

    def _send_sync_to_device(self, device_id: str, sync_data: Dict[str, Any]) -> bool:
        """Send synchronization data to a device"""
        device = self.devices.get(device_id)
        if not device or device.get('status') != 'active':
            return False

        try:
            if device.get('ip_address'):
                url = f"http://{device['ip_address']}:5001/sync"
                response = requests.post(url, json=sync_data, timeout=5)
                return response.status_code == 200
            else:
                # Device-specific sync method (Bluetooth, etc.)
                return self._device_specific_sync(device_id, sync_data)
        except:
            return False

    def discover_devices(self, scan_type: str = "network") -> Dict[str, Any]:
        """Discover new devices in the Ghost ecosystem"""
        discovered_devices = []

        try:
            if scan_type == "network":
                discovered_devices.extend(self._network_device_discovery())
            elif scan_type == "bluetooth":
                discovered_devices.extend(self._bluetooth_device_discovery())
            elif scan_type == "comprehensive":
                discovered_devices.extend(self._network_device_discovery())
                discovered_devices.extend(self._bluetooth_device_discovery())

            # Register discovered devices
            registered_count = 0
            for device in discovered_devices:
                if self._register_device(device):
                    registered_count += 1

            return {
                'success': True,
                'devices_discovered': len(discovered_devices),
                'devices_registered': registered_count,
                'scan_type': scan_type
            }

        except Exception as e:
            return {'success': False, 'error': f'Device discovery failed: {str(e)}'}

    def _network_device_discovery(self) -> List[Dict[str, Any]]:
        """Discover devices on the network"""
        discovered = []

        # Common device ports to check
        device_ports = {
            5001: 'jason_device',  # J.A.S.O.N. device API
            8080: 'smart_home',
            8008: 'roku_tv',
            49153: 'smart_device'
        }

        # Scan local network (simplified)
        base_ip = self._get_local_ip().rsplit('.', 1)[0]

        for i in range(1, 255):
            ip = f"{base_ip}.{i}"
            if ip == self._get_local_ip():
                continue  # Skip self

            for port, device_type in device_ports.items():
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(0.1)
                    result = sock.connect_ex((ip, port))
                    sock.close()

                    if result == 0:
                        discovered.append({
                            'ip_address': ip,
                            'port': port,
                            'type': device_type,
                            'discovery_method': 'port_scan',
                            'capabilities': self._infer_capabilities_from_port(port)
                        })

                except:
                    continue

        return discovered

    def _bluetooth_device_discovery(self) -> List[Dict[str, Any]]:
        """Discover Bluetooth devices"""
        # This would use bluetooth libraries in real implementation
        # Placeholder for Bluetooth device discovery
        return []

    def _infer_capabilities_from_port(self, port: int) -> List[str]:
        """Infer device capabilities from open ports"""
        port_capabilities = {
            5001: ['jason_integration', 'api'],
            8080: ['web_interface', 'streaming'],
            8008: ['media_player', 'remote_control'],
            49153: ['smart_device', 'upnp']
        }
        return port_capabilities.get(port, ['unknown'])

    def _register_device(self, device_info: Dict[str, Any]) -> bool:
        """Register a discovered device in the ecosystem"""
        device_id = device_info.get('device_id') or f"auto_{hashlib.md5(str(device_info).encode()).hexdigest()[:8]}"

        if device_id in self.devices:
            # Update existing device
            self.devices[device_id].update(device_info)
            self.devices[device_id]['last_seen'] = datetime.now().isoformat()
        else:
            # Register new device
            device_info['id'] = device_id
            device_info['status'] = 'active'
            device_info['registered_at'] = datetime.now().isoformat()
            self.devices[device_id] = device_info

            # Initialize device state
            self.device_states[device_id] = {
                'consciousness_active': False,
                'context_loaded': False,
                'last_state': {},
                'pending_transfers': []
            }

        return True

    def transfer_state(self, source_device: str, target_device: str, state_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transfer specific state data between devices"""
        try:
            transfer_id = str(uuid.uuid4())

            transfer_info = {
                'transfer_id': transfer_id,
                'source_device': source_device,
                'target_device': target_device,
                'state_data': state_data,
                'timestamp': datetime.now().isoformat(),
                'status': 'transferring'
            }

            # Execute transfer
            if self._execute_state_transfer(transfer_info):
                transfer_info['status'] = 'completed'
                return {
                    'success': True,
                    'transfer_id': transfer_id,
                    'message': f'State transfer completed: {source_device} → {target_device}'
                }
            else:
                transfer_info['status'] = 'failed'
                return {'success': False, 'error': 'State transfer failed'}

        except Exception as e:
            return {'success': False, 'error': f'State transfer error: {str(e)}'}

    def _execute_state_transfer(self, transfer_info: Dict[str, Any]) -> bool:
        """Execute the actual state transfer"""
        try:
            target_device = transfer_info['target_device']
            state_data = transfer_info['state_data']

            # Compress and send state data
            compressed_data = self._compress_state_data(state_data)
            return self._send_state_to_device(target_device, compressed_data)

        except Exception as e:
            print(f"State transfer execution failed: {e}")
            return False

    def _compress_state_data(self, data: Dict[str, Any]) -> bytes:
        """Compress state data for efficient transfer"""
        import zlib
        json_str = json.dumps(data, default=str)
        return zlib.compress(json_str.encode('utf-8'))

    def _send_state_to_device(self, device_id: str, compressed_data: bytes) -> bool:
        """Send compressed state data to device"""
        device = self.devices.get(device_id)
        if not device or device.get('status') != 'active':
            return False

        try:
            if device.get('ip_address'):
                url = f"http://{device['ip_address']}:5001/state"
                files = {'state_data': compressed_data}
                response = requests.post(url, files=files, timeout=10)
                return response.status_code == 200
            else:
                # Queue for later delivery
                self.sync_queue.put(('state_transfer', device_id, compressed_data))
                return True
        except:
            return False

    def get_ghost_status(self) -> Dict[str, Any]:
        """Get comprehensive Ghost Control status"""
        active_devices = [d for d in self.devices.values() if d.get('status') == 'active']
        conscious_devices = [d for d in self.device_states.values() if d.get('consciousness_active')]

        return {
            'total_devices': len(self.devices),
            'active_devices': len(active_devices),
            'conscious_devices': len(conscious_devices),
            'current_primary': self._get_current_primary_device(),
            'shared_context_active': bool(self.shared_context.get('active_user')),
            'pending_transfers': len(self.state_transfers),
            'last_sync': self.shared_context.get('last_sync'),
            'neural_bridge_active': self.neural_bridge_active,
            'handoff_active': self.handoff_active,
            'device_breakdown': {
                device_type: len([d for d in active_devices if d.get('type') == device_type])
                for device_type in set(d.get('type', 'unknown') for d in active_devices)
            }
        }

    def activate_neural_bridge(self) -> bool:
        """Activate the neural bridge for real-time consciousness synchronization"""
        if self.neural_bridge_active:
            return True

        try:
            self.neural_bridge_active = True
            self.bridge_thread = threading.Thread(target=self._neural_bridge_loop, daemon=True)
            self.bridge_thread.start()
            return True
        except Exception as e:
            print(f"Neural bridge activation failed: {e}")
            self.neural_bridge_active = False
            return False

    def _neural_bridge_loop(self):
        """Main neural bridge synchronization loop"""
        while self.neural_bridge_active:
            try:
                # Synchronize consciousness stream
                self._sync_consciousness_stream()

                # Update device states
                self._update_device_states()

                # Process pending sync queue
                self._process_sync_queue()

                time.sleep(self.sync_interval)

            except Exception as e:
                print(f"Neural bridge loop error: {e}")
                time.sleep(5)

    def _sync_consciousness_stream(self):
        """Synchronize the consciousness stream across devices"""
        # This would continuously sync:
        # - User attention focus
        # - Environmental context
        # - Task progress
        # - Intent predictions
        pass

    def _update_device_states(self):
        """Update and synchronize device states"""
        current_time = datetime.now()

        for device_id, device in self.devices.items():
            if device.get('status') == 'active':
                # Update last seen
                device['last_seen'] = current_time.isoformat()

                # Check for device timeouts
                if self._device_timed_out(device):
                    device['status'] = 'inactive'
                    if self.device_states[device_id]['consciousness_active']:
                        self._handle_device_timeout(device_id)

    def _device_timed_out(self, device: Dict[str, Any]) -> bool:
        """Check if a device has timed out"""
        if not device.get('last_seen'):
            return False

        last_seen = datetime.fromisoformat(device['last_seen'])
        timeout_thresholds = {
            'wearable': timedelta(minutes=5),
            'mobile': timedelta(minutes=10),
            'computer': timedelta(minutes=30),
            'iot': timedelta(hours=1)
        }

        device_type = device.get('type', 'unknown')
        threshold = timeout_thresholds.get(device_type, timedelta(minutes=15))

        return datetime.now() - last_seen > threshold

    def _handle_device_timeout(self, device_id: str):
        """Handle device timeout by transferring consciousness if needed"""
        print(f"Device {device_id} timed out, initiating consciousness preservation...")

        # Find best alternative device
        alternative_device = self._find_best_alternative_device(device_id)

        if alternative_device:
            self.initiate_ghost_handoff(alternative_device, "emergency")
        else:
            # Enter low-power consciousness preservation mode
            self._preserve_consciousness(device_id)

    def _find_best_alternative_device(self, failed_device: str) -> Optional[str]:
        """Find the best alternative device for consciousness transfer"""
        failed_priority = self.device_hierarchy['context_priorities'].get(
            self.devices[failed_device].get('type', 'unknown'), 10
        )

        best_device = None
        best_priority = 10

        for device_id, device in self.devices.items():
            if (device.get('status') == 'active' and
                device_id != failed_device and
                self.device_states[device_id].get('consciousness_active') == False):

                device_priority = self.device_hierarchy['context_priorities'].get(
                    device.get('type', 'unknown'), 10
                )

                if device_priority < best_priority:
                    best_device = device_id
                    best_priority = device_priority

        return best_device

    def _preserve_consciousness(self, device_id: str):
        """Preserve consciousness state when no alternative device is available"""
        print(f"Preserving consciousness for {device_id} in low-power mode")

        # This would save critical state to persistent storage
        # and prepare for restoration when device comes back online

    def _process_sync_queue(self):
        """Process pending synchronization queue"""
        while not self.sync_queue.empty():
            try:
                item = self.sync_queue.get_nowait()
                item_type, device_id, data = item

                if item_type == 'context':
                    self._send_context_to_device(device_id, data)
                elif item_type == 'state_transfer':
                    self._send_state_to_device(device_id, data)

            except queue.Empty:
                break

    def _queue_context_for_later(self, device_id: str, context: Dict[str, Any]):
        """Queue context data for later delivery to offline device"""
        self.sync_queue.put(('context', device_id, context))

    def _send_message_to_device(self, device_id: str, message: Dict[str, Any]):
        """Send a message to a specific device"""
        device = self.devices.get(device_id)
        if device and device.get('ip_address'):
            try:
                url = f"http://{device['ip_address']}:5001/message"
                requests.post(url, json=message, timeout=5)
            except:
                # Queue for later
                self.sync_queue.put(('message', device_id, message))

    def _send_completion_signal(self, source_id: str, target_id: str):
        """Send handoff completion signal to devices"""
        completion_data = {
            'type': 'handoff_completed',
            'source_device': source_id,
            'target_device': target_id,
            'timestamp': datetime.now().isoformat()
        }

        self._send_message_to_device(source_id, completion_data)
        self._send_message_to_device(target_id, completion_data)

        self.hierarchy_status_tool = Tool(
            name="Hierarchy Status",
            description="Get status of master/slave hierarchy",
            func=self.get_hierarchy_status
        )

    def start_consciousness_sync(self):
        """Start the multi-node consciousness synchronization"""
        self.active = True
        self.status_thread = threading.Thread(target=self._status_monitor)
        self.status_thread.daemon = True
        self.status_thread.start()

    def stop_consciousness_sync(self):
        """Stop the multi-node consciousness synchronization"""
        self.active = False
        if self.status_thread:
            self.status_thread.join()

    def register_device(self, device_id: str, ip: str, device_type: str, port: int = 8080):
        """Register a device in the hierarchy"""
        if device_id not in self.hierarchy["capabilities"]:
            return f"Unknown device type: {device_type}"

        self.devices[device_id] = {
            'ip': ip,
            'type': device_type,
            'port': port,
            'status': 'unknown',
            'role': 'master' if device_id == self.master_id else 'slave',
            'capabilities': self.hierarchy["capabilities"].get(device_id, []),
            'priority': self.hierarchy["priorities"].get(device_id, 99),
            'last_seen': time.time(),
            'load': 0.0  # Current load factor
        }

        return f"Registered {device_id} as {'master' if device_id == self.master_id else 'slave'}"

    def delegate_command(self, command: str, params: Dict[str, Any] = None, required_capability: str = None) -> str:
        """Delegate command to appropriate slave based on hierarchy and capabilities"""
        if required_capability:
            # Find slaves with required capability, ordered by priority
            candidates = []
            for device_id, device in self.devices.items():
                if (device['role'] == 'slave' and
                    device['status'] == 'online' and
                    required_capability in device['capabilities']):
                    candidates.append((device_id, device['priority'], device['load']))

            if candidates:
                # Sort by priority (lower number = higher priority), then by load
                candidates.sort(key=lambda x: (x[1], x[2]))
                target_device = candidates[0][0]
                return self.send_command(target_device, command, params)

        # Fallback to master
        return f"Delegated to master: {self.send_command(self.master_id, command, params)}"

    def get_hierarchy_status(self) -> str:
        """Get status of the master/slave hierarchy"""
        status = "J.A.S.O.N. Multi-Node Hierarchy Status:\n\n"
        status += f"Master: {self.master_id}\n"
        status += "Slaves:\n"

        for slave_id in self.hierarchy["slaves"]:
            if slave_id in self.devices:
                device = self.devices[slave_id]
                status += f"  • {slave_id} ({device['type']}): {device['status']} - Priority {device['priority']}\n"
                status += f"    Capabilities: {', '.join(device['capabilities'])}\n"
                status += f"    Load: {device['load']:.1f}\n"
            else:
                status += f"  • {slave_id}: Not registered\n"

        status += "\nActive Connections:\n"
        for device_id, device in self.devices.items():
            if device['status'] == 'online':
                status += f"  • {device_id} at {device['ip']}:{device['port']}\n"

        return status

    def balance_load(self):
        """Balance load across slave devices"""
        online_slaves = [d for d in self.devices.values() if d['role'] == 'slave' and d['status'] == 'online']
        if len(online_slaves) > 1:
            avg_load = sum(d['load'] for d in online_slaves) / len(online_slaves)
            # In a real implementation, migrate tasks from high-load to low-load devices
            pass

    def failover_check(self):
        """Check for device failures and initiate failover"""
        for device_id, device in list(self.devices.items()):
            if device['status'] == 'offline' and time.time() - device['last_seen'] > 300:  # 5 minutes
                # Attempt to find backup device with same capabilities
                backup_candidates = []
                for backup_id, backup_device in self.devices.items():
                    if (backup_device['status'] == 'online' and
                        set(device['capabilities']).issubset(set(backup_device['capabilities'])) and
                        backup_device['role'] == 'slave'):
                        backup_candidates.append(backup_id)

                if backup_candidates:
                    # Migrate critical tasks to backup
                    pass

    def discover_devices(self, network_range: str = "192.168.1.0/24") -> str:
        """Discover devices on the network"""
        # Simple discovery - in real implementation, use nmap or similar
        discovered = []

        # For demo, add known devices
        # In production, scan network for devices with J.A.S.O.N. endpoints

        if discovered:
            return f"Discovered devices: {', '.join(discovered)}"
        else:
            return "No new devices discovered. Manual registration required."

    def send_command(self, device_id: str, command: str, params: Dict[str, Any] = None) -> str:
        """Send a command to a device"""
        if device_id not in self.devices:
            return f"Device {device_id} not registered"

        device = self.devices[device_id]
        if device['status'] != 'online':
            return f"Device {device_id} is not online"

        try:
            url = f"http://{device['ip']}:{device['port']}/command"
            payload = {
                'command': command,
                'params': params or {},
                'timestamp': time.time()
            }

            response = requests.post(url, json=payload, timeout=5)
            if response.status_code == 200:
                result = response.json()
                return f"Command sent to {device_id}: {result.get('response', 'OK')}"
            else:
                return f"Command failed: {response.status_code}"

        except requests.RequestException as e:
            device['status'] = 'offline'
            return f"Command failed: {e}"

    def get_device_status(self) -> str:
        """Get status of all devices"""
        status_report = []
        for device_id, device in self.devices.items():
            status_report.append(f"{device_id} ({device['type']}): {device['status']} at {device['ip']}")

        if status_report:
            return "Device Status:\n" + "\n".join(status_report)
        else:
            return "No devices registered"

    def relay_conversation(self, message: str, source_device: str = "pc"):
        """Relay conversation to other devices for continuity"""
        for device_id, device in self.devices.items():
            if device_id != source_device and device['status'] == 'online':
                self.send_command(device_id, "conversation_update", {"message": message})

    def _status_monitor(self):
        """Monitor device status in background"""
        while self.active:
            for device_id, device in list(self.devices.items()):
                try:
                    url = f"http://{device['ip']}:{device['port']}/status"
                    response = requests.get(url, timeout=2)
                    if response.status_code == 200:
                        device['status'] = 'online'
                        device['last_seen'] = time.time()
                    else:
                        device['status'] = 'offline'
                except requests.RequestException:
                    device['status'] = 'offline'

            time.sleep(30)  # Check every 30 seconds

    def execute_cross_device_task(self, task: str) -> str:
        """Execute a task that may span multiple devices"""
        # Determine which devices are best for the task
        if "mobile" in task.lower() or "location" in task.lower():
            # Use phone for location-based tasks
            phone_devices = [d for d in self.devices.values() if d['type'] == 'phone' and d['status'] == 'online']
            if phone_devices:
                return self.send_command(list(self.devices.keys())[list(self.devices.values()).index(phone_devices[0])], "execute_task", {"task": task})

        elif "sensor" in task.lower() or "environment" in task.lower():
            # Use ESP32 for sensor tasks
            esp_devices = [d for d in self.devices.values() if d['type'] == 'esp32' and d['status'] == 'online']
            if esp_devices:
                return self.send_command(list(self.devices.keys())[list(self.devices.values()).index(esp_devices[0])], "execute_task", {"task": task})

        # Default to PC
        return "Task executed on primary device"
