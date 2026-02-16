"""
J.A.S.O.N. Ghost in the Machine - Full OS Mastery Module
"""

import os
import subprocess
import psutil
import requests
from langchain_core.tools import Tool
import webbrowser
import pyautogui
import keyboard
from typing import Dict, Any, List

# Spectrum Command - Broadlink IR/RF support
broadlink_available = False
try:
    import broadlink
    broadlink_available = True
except ImportError:
    pass

class OSManager:
    def __init__(self):
        # Create tools
        self.file_operations_tool = Tool(
            name="File Operations",
            func=self.file_operations,
            description="Perform file system operations: read, write, move, delete files"
        )

        self.web_search_tool = Tool(
            name="Web Search",
            func=self.web_search,
            description="Search the web for information"
        )

        self.automation_tool = Tool(
            name="System Automation",
            func=self.system_automation,
            description="Automate system tasks, open applications, control media"
        )

        self.network_scan_tool = Tool(
            name="Network Scan",
            func=self.network_scan,
            description="Scan local network for devices and security"
        )

        self.iot_control_tool = Tool(
            name="IoT Control",
            func=self.iot_control,
            description="Control IoT devices: lights, thermostat, locks"
        )

        # Hard Integration tools
        self.home_assistant_tool = Tool(
            name="Home Assistant Control",
            func=self.home_assistant_control,
            description="Control IoT devices through Home Assistant REST API"
        )

        self.web_automation_tool = Tool(
            name="Web Automation",
            func=self.web_automation,
            description="Automate web interactions using Playwright browser control"
        )

        self.kernel_control_tool = Tool(
            name="Kernel Control",
            func=self.kernel_control,
            description="Execute kernel-level commands and scripts"
        )

        # Guard Dog tools
        self.guard_dog_tool = Tool(
            name="Network Security",
            func=self.guard_dog_scan,
            description="Monitor network for intrusions and suspicious activity"
        )

        self.block_connection_tool = Tool(
            name="Block Connection",
            func=self.block_connection,
            description="Block suspicious network connections"
        )

        # Financial Advisor tools
        self.financial_analysis_tool = Tool(
            name="Financial Analysis",
            func=self.analyze_finances,
            description="Analyze bank statements and provide financial advice"
        )

        self.crypto_tracking_tool = Tool(
            name="Crypto Tracking",
            func=self.track_crypto,
            description="Monitor cryptocurrency prices and suggest trades"
        )

        # Spectrum Command tools
        self.spectrum_command_tool = Tool(
            name="Spectrum Command",
            func=self.spectrum_command,
            description="Control legacy hardware via IR/RF signals using Broadlink RM4"
        )

        # Optional dependencies
        self.playwright_available = False
        try:
            from playwright.sync_api import sync_playwright
            self.playwright_available = True
        except ImportError:
            pass

        # Broadlink devices
        self.broadlink_devices = []
        if broadlink_available:
            try:
                self.broadlink_devices = broadlink.discover(timeout=5)
            except:
                pass

    def file_operations(self, operation: str, path: str = "", content: str = "", query: str = "") -> str:
        """Perform file operations with proactive navigation"""
        try:
            if operation == "read" and path:
                with open(path, 'r') as f:
                    return f.read()
            elif operation == "write" and path and content:
                with open(path, 'w') as f:
                    f.write(content)
                return f"Written to {path}"
            elif operation == "list" and path:
                items = os.listdir(path)
                return "\n".join(items)
            elif operation == "find" and query:
                # Proactive file-tree navigation
                return self._proactive_file_search(query)
            elif operation == "delete" and path:
                os.remove(path)
                return f"Deleted {path}"
            else:
                return "Invalid file operation"
        except Exception as e:
            return f"File operation failed: {e}"

    def _proactive_file_search(self, query: str) -> str:
        """Proactive file search with categorization"""
        import glob
        import time
        from pathlib import Path

        # Search in common directories
        search_paths = [
            os.path.expanduser("~/Desktop"),
            os.path.expanduser("~/Documents"),
            os.path.expanduser("~/Downloads"),
            os.path.expanduser("~/Pictures"),
            os.path.expanduser("~/Videos"),
            "/tmp"  # For temporary files
        ]

        found_files = []
        for search_path in search_paths:
            if os.path.exists(search_path):
                # Recursive search with glob
                pattern = os.path.join(search_path, "**", f"*{query}*")
                matches = glob.glob(pattern, recursive=True)
                for match in matches[:50]:  # Limit results
                    if os.path.isfile(match):
                        stat = os.stat(match)
                        found_files.append({
                            "path": match,
                            "size": stat.st_size,
                            "modified": stat.st_mtime,
                            "category": self._categorize_file(match, search_path)
                        })

        if not found_files:
            return f"No files found matching '{query}'"

        # Sort by relevance (recently modified first)
        found_files.sort(key=lambda x: x["modified"], reverse=True)

        # Group by category
        categories = {}
        for file_info in found_files[:20]:  # Top 20 results
            cat = file_info["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(file_info)

        # Format response
        response = f"Proactive File Search Results for '{query}':\n\n"
        for category, files in categories.items():
            response += f"{category.upper()}:\n"
            for file_info in files:
                mod_time = time.strftime("%Y-%m-%d %H:%M", time.localtime(file_info["modified"]))
                size_mb = file_info["size"] / (1024 * 1024)
                response += f"  • {os.path.basename(file_info['path'])} ({size_mb:.1f} MB, {mod_time})\n"
                response += f"    Path: {file_info['path']}\n"
            response += "\n"

        response += "Lead, which file should I initialize?"
        return response

    def _categorize_file(self, filepath: str, search_path: str) -> str:
        """Categorize file based on location and type"""
        basename = os.path.basename(filepath).lower()
        dirname = os.path.dirname(filepath)

        # Categorize by location
        if "desktop" in dirname.lower():
            return "Local Draft"
        elif "downloads" in dirname.lower():
            return "Downloaded"
        elif "documents" in dirname.lower():
            return "Document"
        elif "pictures" in dirname.lower() or "images" in dirname.lower():
            return "Visual Media"
        elif "videos" in dirname.lower():
            return "Video Media"
        elif "/tmp" in dirname:
            return "Temporary"
        else:
            # Check modification time
            stat = os.stat(filepath)
            days_old = (time.time() - stat.st_mtime) / (24 * 3600)
            if days_old < 1:
                return "Recently Edited"
            elif days_old < 7:
                return "This Week"
            elif days_old < 30:
                return "This Month"
            else:
                return "Archived"

    def web_search(self, query: str) -> str:
        """Search the web"""
        try:
            # Use DuckDuckGo or similar for search
            url = f"https://duckduckgo.com/html/?q={query}"
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            # Simple parsing - in real implementation, use BeautifulSoup
            if response.status_code == 200:
                return f"Search results for '{query}': {response.text[:1000]}..."
            else:
                return f"Search failed: {response.status_code}"
        except Exception as e:
            return f"Web search failed: {e}"

    def system_automation(self, action: str, target: str = "") -> str:
        """Automate system tasks"""
        try:
            if action == "open_app" and target:
                if os.name == 'nt':  # Windows
                    subprocess.Popen([target])
                else:  # macOS/Linux
                    subprocess.Popen([target])
                return f"Opened {target}"

            elif action == "open_url" and target:
                webbrowser.open(target)
                return f"Opened {target}"

            elif action == "play_pause":
                pyautogui.press('playpause')
                return "Toggled media playback"

            elif action == "volume_up":
                pyautogui.press('volumeup')
                return "Increased volume"

            elif action == "volume_down":
                pyautogui.press('volumedown')
                return "Decreased volume"

            elif action == "type_text" and target:
                pyautogui.write(target)
                return f"Typed: {target}"

            elif action == "press_key" and target:
                pyautogui.press(target)
                return f"Pressed key: {target}"

            else:
                return "Unknown automation action"

        except Exception as e:
            return f"Automation failed: {e}"

    def network_scan(self) -> str:
        """Scan local network for security"""
        try:
            # Get network info
            net_info = psutil.net_if_addrs()
            connections = psutil.net_connections()

            result = "Network Information:\n"
            for interface, addrs in net_info.items():
                result += f"Interface: {interface}\n"
                for addr in addrs:
                    result += f"  {addr.family.name}: {addr.address}\n"

            result += "\nActive Connections:\n"
            for conn in connections[:10]:  # Limit to 10
                result += f"  {conn.laddr} -> {conn.raddr} ({conn.status})\n"

            return result

        except Exception as e:
            return f"Network scan failed: {e}"

    def iot_control(self, device: str, command: str) -> str:
        """Control IoT devices"""
        # This would integrate with specific IoT platforms
        # For demo, simulate basic controls

        iot_devices = {
            "lights": {"on": "Lights turned on", "off": "Lights turned off"},
            "thermostat": {"up": "Temperature increased", "down": "Temperature decreased"},
            "lock": {"lock": "Door locked", "unlock": "Door unlocked"}
        }

        if device in iot_devices and command in iot_devices[device]:
            # In real implementation, send commands to IoT hub
            return iot_devices[device][command]
        else:
            return f"Unknown IoT command: {device} {command}"

    def home_assistant_control(self, entity_id: str, action: str, ha_url: str = "http://homeassistant.local:8123", token: str = "") -> str:
        """Control IoT devices through Home Assistant REST API"""
        try:
            if not token:
                return "Home Assistant token required for authentication"

            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }

            # Determine service call based on action
            if action in ["turn_on", "turn_off"]:
                service = action
                domain = "light" if "light" in entity_id else "switch"
            elif action in ["lock", "unlock"]:
                service = action
                domain = "lock"
            elif action in ["set_temperature"]:
                service = "set_temperature"
                domain = "climate"
            else:
                return f"Unsupported action: {action}"

            url = f"{ha_url}/api/services/{domain}/{service}"
            payload = {"entity_id": entity_id}

            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                return f"Successfully executed {action} on {entity_id}"
            else:
                return f"Home Assistant error: {response.status_code} - {response.text}"

        except Exception as e:
            return f"Home Assistant control failed: {e}"

    def web_automation(self, action: str, url: str = "", selector: str = "", text: str = "") -> str:
        """Automate web interactions using Playwright"""
        if not self.playwright_available:
            return "Playwright not available - install with: pip install playwright && playwright install"

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=False)  # Visible browser for interaction
                page = browser.new_page()

                if action == "navigate" and url:
                    page.goto(url)
                    return f"Navigated to {url}"

                elif action == "click" and selector:
                    if url:
                        page.goto(url)
                    page.click(selector)
                    return f"Clicked element: {selector}"

                elif action == "fill" and selector and text:
                    if url:
                        page.goto(url)
                    page.fill(selector, text)
                    return f"Filled {selector} with: {text}"

                elif action == "screenshot" and url:
                    page.goto(url)
                    screenshot_path = "/tmp/jason_screenshot.png"
                    page.screenshot(path=screenshot_path)
                    return f"Screenshot saved to {screenshot_path}"

                else:
                    return f"Unknown web automation action: {action}"

                browser.close()

        except Exception as e:
            return f"Web automation failed: {e}"

    def kernel_control(self, command: str, script_path: str = "", args: List[str] = None) -> str:
        """Execute kernel-level commands and scripts"""
        try:
            if script_path:
                # Execute script file
                if os.path.exists(script_path):
                    if script_path.endswith('.sh'):
                        cmd = ['bash', script_path]
                    elif script_path.endswith('.bat'):
                        cmd = ['cmd', '/c', script_path]
                    elif script_path.endswith('.py'):
                        cmd = ['python3', script_path]
                    else:
                        return f"Unsupported script type: {script_path}"

                    if args:
                        cmd.extend(args)

                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=60,
                        cwd=os.path.dirname(script_path) if os.path.dirname(script_path) else None
                    )

                    if result.returncode == 0:
                        return f"Script executed successfully: {result.stdout}"
                    else:
                        return f"Script failed: {result.stderr}"

                else:
                    return f"Script not found: {script_path}"

            elif command:
                # Execute direct command
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if result.returncode == 0:
                    return f"Command executed: {result.stdout}"
                else:
                    return f"Command failed: {result.stderr}"

            else:
                return "No command or script specified"

        except subprocess.TimeoutExpired:
            return "Command timed out"
        except Exception as e:
            return f"Kernel control failed: {e}"

    def guard_dog_scan(self) -> str:
        """Monitor network for intrusions and suspicious activity"""
        try:
            connections = psutil.net_connections()
            suspicious_activity = []
            connection_counts = {}

            # Analyze connections
            for conn in connections:
                if conn.raddr:
                    remote_ip = conn.raddr.ip
                    remote_port = conn.raddr.port

                    # Count connections per IP
                    if remote_ip not in connection_counts:
                        connection_counts[remote_ip] = 0
                    connection_counts[remote_ip] += 1

                    # Check for suspicious ports
                    suspicious_ports = [22, 23, 3389, 5900, 6667]  # SSH, Telnet, RDP, VNC, IRC
                    if remote_port in suspicious_ports:
                        suspicious_activity.append(f"Suspicious service on port {remote_port} from {remote_ip}")

                    # Check for unusual high ports or known malicious
                    if remote_port > 49151 and remote_port not in [80, 443, 53, 25, 110, 143, 993, 995]:
                        suspicious_activity.append(f"Unusual high port {remote_port} from {remote_ip}")

            # Check for IPs with too many connections
            for ip, count in connection_counts.items():
                if count > 10:  # Arbitrary threshold
                    suspicious_activity.append(f"High connection count ({count}) from {ip}")

            # Check for listening ports that shouldn't be open
            listening_ports = [conn.laddr.port for conn in connections if conn.status == 'LISTEN']
            dangerous_ports = [21, 23, 25, 53, 110, 143]  # FTP, Telnet, SMTP, DNS, POP3, IMAP
            for port in listening_ports:
                if port in dangerous_ports:
                    suspicious_activity.append(f"Dangerous service listening on port {port}")

            if suspicious_activity:
                report = "GUARD DOG ALERT - Suspicious activity detected:\n" + "\n".join(suspicious_activity)
                # Auto-block most severe threats
                for activity in suspicious_activity:
                    if "Suspicious service" in activity or "High connection count" in activity:
                        ip = activity.split()[-1]  # Extract IP
                        self._auto_block_ip(ip)
                        report += f"\nAUTO-BLOCKED: {ip}"
                return report
            else:
                return "Network scan complete - No suspicious activity detected"

        except Exception as e:
            return f"Guard dog scan failed: {e}"

    def block_connection(self, ip_address: str, permanent: bool = False) -> str:
        """Block a network connection"""
        try:
            if os.name == 'nt':  # Windows
                # Use Windows Firewall
                cmd = f'netsh advfirewall firewall add rule name="JASON Block {ip_address}" dir=in action=block remoteip={ip_address}'
                if permanent:
                    cmd += ' permanent=yes'
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            else:  # macOS/Linux
                # Use pfctl on macOS or iptables on Linux
                if os.uname().sysname == 'Darwin':  # macOS
                    # Add to pf.conf temporarily
                    rule = f'block drop from {ip_address} to any\n'
                    with open('/etc/pf.conf', 'a') as f:
                        f.write(rule)
                    subprocess.run(['pfctl', '-f', '/etc/pf.conf'], capture_output=True)
                    subprocess.run(['pfctl', '-e'], capture_output=True)  # Enable if not enabled
                else:  # Linux
                    subprocess.run(['iptables', '-A', 'INPUT', '-s', ip_address, '-j', 'DROP'], capture_output=True)

            return f"Blocked connection from {ip_address}"

        except Exception as e:
            return f"Failed to block connection: {e}"

    def _auto_block_ip(self, ip: str):
        """Automatically block suspicious IP"""
        try:
            self.block_connection(ip, permanent=False)
        except:
            pass  # Silent fail for auto-blocking

    def analyze_finances(self, statement_path: str = "", risk_profile: str = "moderate") -> str:
        """Analyze bank statements and provide financial advice"""
        try:
            transactions = []

            if statement_path and os.path.exists(statement_path):
                # Read actual statement file (CSV format assumed)
                with open(statement_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines[1:]:  # Skip header
                        parts = line.strip().split(',')
                        if len(parts) >= 3:
                            date, description, amount = parts[0], parts[1], float(parts[2])
                            transactions.append({
                                'date': date,
                                'description': description,
                                'amount': amount
                            })
            else:
                # Generate mock transactions for demo
                import random
                from datetime import datetime, timedelta

                categories = ['Grocery', 'Utilities', 'Entertainment', 'Salary', 'Investment', 'Dining']
                for i in range(50):
                    date = (datetime.now() - timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d')
                    description = random.choice(categories)
                    amount = random.uniform(-500, 2000) if description != 'Salary' else random.uniform(3000, 5000)
                    transactions.append({
                        'date': date,
                        'description': description,
                        'amount': amount
                    })

            # Analyze transactions
            total_income = sum(t['amount'] for t in transactions if t['amount'] > 0)
            total_expenses = abs(sum(t['amount'] for t in transactions if t['amount'] < 0))

            # Categorize expenses
            expense_categories = {}
            for t in transactions:
                if t['amount'] < 0:
                    cat = t['description']
                    if cat not in expense_categories:
                        expense_categories[cat] = 0
                    expense_categories[cat] += abs(t['amount'])

            # Generate advice based on risk profile
            advice = []
            savings_rate = (total_income - total_expenses) / total_income if total_income > 0 else 0

            if savings_rate < 0.1:
                advice.append("Increase savings rate above 10% of income")
            elif savings_rate > 0.3:
                advice.append("Excellent savings rate - consider investment opportunities")

            if risk_profile == "conservative":
                advice.append("Focus on low-risk investments like bonds and CDs")
            elif risk_profile == "moderate":
                advice.append("Balanced portfolio: 60% stocks, 40% bonds recommended")
            elif risk_profile == "aggressive":
                advice.append("High-risk, high-reward strategy: heavy crypto and tech stock allocation")

            # Spending analysis
            top_expenses = sorted(expense_categories.items(), key=lambda x: x[1], reverse=True)[:3]
            if top_expenses:
                advice.append(f"Top spending categories: {', '.join(f'{cat}: ${amt:.2f}' for cat, amt in top_expenses)}")

            report = f"""
FINANCIAL ANALYSIS REPORT

Income: ${total_income:.2f}
Expenses: ${total_expenses:.2f}
Net Savings: ${total_income - total_expenses:.2f}
Savings Rate: {savings_rate:.1%}

Expense Breakdown:
{chr(10).join(f"- {cat}: ${amt:.2f}" for cat, amt in expense_categories.items())}

Recommendations:
{chr(10).join(f"- {rec}" for rec in advice)}
"""

            return report.strip()

        except Exception as e:
            return f"Financial analysis failed: {e}"

    def track_crypto(self, symbols: List[str] = None, risk_profile: str = "moderate") -> str:
        """Monitor cryptocurrency prices and suggest trades"""
        try:
            if symbols is None:
                symbols = ['bitcoin', 'ethereum', 'solana', 'cardano']

            # Use CoinGecko API (free tier)
            base_url = "https://api.coingecko.com/api/v3"
            prices = {}

            for symbol in symbols:
                try:
                    response = requests.get(f"{base_url}/simple/price", params={
                        'ids': symbol,
                        'vs_currencies': 'usd',
                        'include_24hr_change': 'true'
                    }, timeout=5)

                    if response.status_code == 200:
                        data = response.json()
                        if symbol in data:
                            prices[symbol] = {
                                'price': data[symbol]['usd'],
                                'change_24h': data[symbol]['usd_24h_change']
                            }
                except:
                    continue

            if not prices:
                return "Unable to fetch cryptocurrency data"

            # Generate trading suggestions based on risk profile
            suggestions = []

            for symbol, data in prices.items():
                price = data['price']
                change = data['change_24h']

                if risk_profile == "conservative":
                    if change > 5:
                        suggestions.append(f"SELL {symbol.upper()}: Price up {change:.2f}% - take profits")
                    elif change < -10:
                        suggestions.append(f"BUY {symbol.upper()}: Price down {change:.2f}% - potential rebound")
                    else:
                        suggestions.append(f"HOLD {symbol.upper()}: Stable at ${price:.2f}")

                elif risk_profile == "moderate":
                    if change > 10:
                        suggestions.append(f"SELL {symbol.upper()}: Strong uptrend {change:.2f}%")
                    elif change < -5:
                        suggestions.append(f"BUY {symbol.upper()}: Dip detected {change:.2f}%")
                    else:
                        suggestions.append(f"HOLD {symbol.upper()}: ${price:.2f}")

                elif risk_profile == "aggressive":
                    if change > 0:
                        suggestions.append(f"BUY MORE {symbol.upper()}: Momentum positive {change:.2f}%")
                    else:
                        suggestions.append(f"HODL {symbol.upper()}: Temporary dip ${price:.2f}")

            report = f"""
CRYPTO MARKET REPORT

Current Prices:
{chr(10).join(f"- {symbol.upper()}: ${data['price']:.2f} ({data['change_24h']:+.2f}%)" for symbol, data in prices.items())}

Trading Recommendations ({risk_profile.title()} Risk):
{chr(10).join(f"- {suggestion}" for suggestion in suggestions)}

Risk Profile Notes:
- Conservative: Focus on stability and profit-taking
- Moderate: Balance growth with risk management
- Aggressive: Maximize gains, tolerate high volatility
"""

            return report.strip()

        except Exception as e:
            return f"Crypto tracking failed: {e}"

    def spectrum_command(self, device_type: str, command: str) -> str:
        """Control legacy hardware via IR/RF signals"""
        if not broadlink_available:
            return "Broadlink library not available"

        if not self.broadlink_devices:
            return "No Broadlink devices discovered"

        try:
            device = self.broadlink_devices[0]  # Use first discovered device

            # Predefined IR/RF codes for common devices
            # In practice, these would be learned from the actual remote
            codes = {
                "ac_22": b'\x26\x00\x00\x00\x00\x00\x00\x00',  # AC set to 22°C
                "ac_on": b'\x26\x00\x01\x00\x00\x00\x00\x00',   # AC power on
                "ac_off": b'\x26\x00\x02\x00\x00\x00\x00\x00',  # AC power off
                "tv_power": b'\x20\x00\x00\x00\x00\x00\x00\x00', # TV power
                "tv_volume_up": b'\x20\x00\x01\x00\x00\x00\x00\x00', # TV volume up
                "tv_volume_down": b'\x20\x00\x02\x00\x00\x00\x00\x00' # TV volume down
            }

            code_key = f"{device_type}_{command}".lower()
            if code_key in codes:
                device.send_data(codes[code_key])
                return f"IR/RF command '{command}' sent to {device_type}"
            else:
                return f"No predefined code for {device_type} {command}. Use learn_code to capture."

        except Exception as e:
            return f"Spectrum command failed: {e}"

    def learn_ir_code(self, device_type: str, command: str) -> str:
        """Learn a new IR code from remote"""
        if not broadlink_available or not self.broadlink_devices:
            return "Broadlink not available or no devices"

        try:
            device = self.broadlink_devices[0]
            device.enter_learning()

            # Wait for user to press button on remote
            import time
            time.sleep(10)  # Give user time

            data = device.check_data()
            if data:
                # Save the code (in practice, store in database)
                return f"IR code learned for {device_type} {command}"
            else:
                return "No IR code received"

        except Exception as e:
            return f"IR learning failed: {e}"
