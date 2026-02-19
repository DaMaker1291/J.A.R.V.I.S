"""
J.A.S.O.N. General Assistant Module
AI-Powered Task Execution for Any Request
"""

import logging
import json
from typing import Dict, Any, List, Optional
import google.generativeai as genai
from jason.modules.desktop_control import DesktopController
from jason.modules.concierge import ConciergeManager
from jason.tools.browser_agent import BrowserAgent
from jason.tools.vpn_control import VPNController
from jason.modules.dropshipping_automation import DropshippingAutomation

logger = logging.getLogger(__name__)

class GeneralAssistant:
    """AI-powered assistant that can handle any user request"""

    def __init__(self, config: dict):
        self.config = config

        # Initialize Gemini AI
        genai.configure(api_key=config.get('gemini_api_key'))
        self.model = genai.GenerativeModel('gemini-1.5-flash')

        # Initialize tools
        self.desktop = DesktopController(config.get('desktop', {}))
        self.concierge = ConciergeManager(config)
        self.browser = BrowserAgent(captcha_api_key=config.get('captcha_api_key'))
        self.vpn = VPNController(vpn_provider=config.get('vpn', {}).get('provider', 'nordvpn'))
        self.dropshipping = DropshippingAutomation(config)

        # Request history for context
        self.conversation_history = []

    def execute_request(self, request: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute any user request using AI planning and available tools"""

        context = context or {}

        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": request})

        try:
            # Step 1: Classify and plan the request
            plan = self._create_execution_plan(request, context)

            if not plan.get('executable', False):
                # Need more information
                clarification_question = plan.get('clarification_question')
                return {
                    'success': False,
                    'needs_clarification': True,
                    'question': clarification_question,
                    'message': 'Need more information to proceed'
                }

            # Step 2: Execute the plan
            result = self._execute_plan(plan)

            # Step 3: Format response
            response = self._format_response(result, request)

            # Add to history
            self.conversation_history.append({"role": "assistant", "content": response})

            return {
                'success': True,
                'response': response,
                'executed_steps': len(plan.get('steps', [])),
                'tools_used': plan.get('tools_used', [])
            }

        except Exception as e:
            logger.error(f"Request execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to execute request'
            }

    def _create_execution_plan(self, request: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Use AI to create an execution plan for the request, with fallback for missing API key"""

        # Check if Gemini key is available
        if not self.config.get('gemini_api_key'):
            return self._fallback_execution_plan(request, context)

        prompt = f"""
You are J.A.S.O.N., an AI assistant capable of executing any user request using various tools and automation.

Available Tools:
- desktop_control: Mouse/keyboard automation, open/close apps, human-like typing
- browser_agent: Web browsing, form filling, captcha handling
- concierge: Booking systems, travel planning with arbitrage
- vpn_control: IP location changing for price comparison
- dropshipping: E-commerce automation for dropshipping businesses
- general_ai: Direct AI responses and analysis

User Request: {request}

Context: {json.dumps(context)}

Analyze the request and create a detailed execution plan. Consider:
1. What is the user trying to accomplish?
2. What tools are needed?
3. What steps should be taken?
4. Are there any safety considerations?
5. Do we need more information from the user?

If more information is needed, set executable to false and provide a clarification question.

Otherwise, provide:
- executable: true
- category: (desktop, web, booking, general, etc.)
- tools_used: list of tools needed
- steps: detailed step-by-step plan
- safety_checks: any safety considerations
- estimated_duration: rough time estimate

Respond in JSON format only.
"""

        try:
            response = self.model.generate_content(prompt)
            plan_text = response.text.strip()

            # Clean JSON if needed
            if plan_text.startswith('```json'):
                plan_text = plan_text[7:]
            if plan_text.endswith('```'):
                plan_text = plan_text[:-3]

            plan = json.loads(plan_text)
            return plan

        except Exception as e:
            logger.error(f"Plan creation failed: {e}")
            return self._fallback_execution_plan(request, context)

    def _fallback_execution_plan(self, request: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback planning using rule-based classification when AI is not available"""
        request_lower = request.lower()

        # Open application requests
        if "open" in request_lower and "teams" in request_lower:
            return {
                "executable": True,
                "category": "desktop",
                "tools_used": ["desktop_control"],
                "steps": [{
                    "type": "desktop_action",
                    "params": {"action": "open_app", "app_name": "Microsoft Teams"}
                }],
                "safety_checks": ["Ensure Teams is installed"],
                "estimated_duration": 5
            }

        # Typing requests
        elif "type" in request_lower and ("essay" in request_lower or "text" in request_lower):
            text_to_type = request.split("type", 1)[1].strip() if "type" in request else "Sample essay text"
            return {
                "executable": True,
                "category": "desktop",
                "tools_used": ["desktop_control"],
                "steps": [{
                    "type": "desktop_action",
                    "params": {"action": "type_text", "text": text_to_type, "human_like": True}
                }],
                "safety_checks": ["Ensure cursor is in correct input field"],
                "estimated_duration": 30
            }

        # Homework requests
        elif "homework" in request_lower or "hwk" in request_lower:
            return {
                "executable": True,
                "category": "desktop",
                "tools_used": ["desktop_control"],
                "steps": [
                    {
                        "type": "desktop_action",
                        "params": {"action": "open_app", "app_name": "Microsoft Teams"}
                    },
                    {
                        "type": "wait",
                        "params": {"seconds": 3}
                    },
                    {
                        "type": "desktop_action",
                        "params": {"action": "type_text", "text": "Checking for homework assignments...", "human_like": True}
                    }
                ],
                "safety_checks": ["Ensure Teams is properly configured"],
                "estimated_duration": 10
            }

        # Screenshot requests
        elif "screenshot" in request_lower or "screen" in request_lower:
            return {
                "executable": True,
                "category": "desktop",
                "tools_used": ["desktop_control"],
                "steps": [{
                    "type": "desktop_action",
                    "params": {"action": "take_screenshot"}
                }],
                "safety_checks": [],
                "estimated_duration": 2
            }

        # Mouse movement requests
        elif "mouse" in request_lower and ("move" in request_lower or "position" in request_lower):
            # Extract coordinates if provided
            import re
            coords = re.findall(r'\d+', request)
            x, y = int(coords[0]) if len(coords) > 0 else 500, int(coords[1]) if len(coords) > 1 else 500
            return {
                "executable": True,
                "category": "desktop",
                "tools_used": ["desktop_control"],
                "steps": [{
                    "type": "desktop_action",
                    "params": {"action": "move_mouse", "x": x, "y": y, "duration": 1.0}
                }],
                "safety_checks": ["Coordinates within safe bounds"],
                "estimated_duration": 2
            }

        # Booking requests
        elif "book" in request_lower and ("holiday" in request_lower or "trip" in request_lower or "flight" in request_lower):
            destination = "Japan" if "japan" in request_lower else "Unknown"
            return {
                "executable": True,
                "category": "booking",
                "tools_used": ["concierge", "browser_agent"],
                "steps": [{
                    "type": "booking_action",
                    "params": {
                        "booking_request": {
                            "type": "flight",
                            "destination": destination,
                            "date": "2024-03-15"  # Default date
                        }
                    }
                }],
                "safety_checks": ["Valid booking details"],
                "estimated_duration": 60
            }

        # Dropshipping requests
        elif "dropshipping" in request_lower or "automate" in request_lower and "business" in request_lower:
            return {
                "executable": True,
                "category": "ecommerce",
                "tools_used": ["dropshipping", "browser_agent"],
                "steps": [
                    {
                        "type": "dropshipping_action",
                        "params": {"action": "check_orders", "platform": "shopify", "credentials": {}}
                    },
                    {
                        "type": "dropshipping_action",
                        "params": {"action": "update_inventory", "platform": "shopify", "credentials": {}, "products": []}
                    }
                ],
                "safety_checks": ["Valid platform credentials"],
                "estimated_duration": 30
            }

        # Search requests
        elif "search" in request_lower or "find" in request_lower:
            query = request.split("search", 1)[1].strip() if "search" in request else request
            return {
                "executable": True,
                "category": "web",
                "tools_used": ["browser_agent"],
                "steps": [{
                    "type": "browser_action",
                    "params": {"action": "navigate", "url": f"https://www.google.com/search?q={query.replace(' ', '+')}"}
                }],
                "safety_checks": ["Safe search query"],
                "estimated_duration": 10
            }

        # Default fallback
        else:
            return {
                "executable": False,
                "error": "Unable to understand request without AI assistance",
                "clarification_question": "Could you please clarify what you want me to do? For example: 'open Teams', 'type an essay', 'book a flight to Japan', 'take a screenshot', 'check orders'."
            }

    def _execute_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the planned steps"""

        results = []
        steps = plan.get('steps', [])

        for step in steps:
            try:
                step_result = self._execute_step(step)
                results.append(step_result)

                # Check if step failed critically
                if step_result.get('critical_failure', False):
                    break

            except Exception as e:
                logger.error(f"Step execution failed: {e}")
                results.append({
                    'step': step,
                    'success': False,
                    'error': str(e)
                })

        return {
            'steps_executed': results,
            'overall_success': all(r.get('success', False) for r in results)
        }

    def _execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single step using appropriate tools"""

        step_type = step.get('type')
        params = step.get('params', {})

        try:
            if step_type == 'desktop_action':
                return self._execute_desktop_action(params)
            elif step_type == 'browser_action':
                return self._execute_browser_action(params)
            elif step_type == 'booking_action':
                return self._execute_booking_action(params)
            elif step_type == 'vpn_action':
                return self._execute_vpn_action(params)
            elif step_type == 'ai_response':
                return self._execute_ai_response(params)
            elif step_type == 'dropshipping_action':
                return self._execute_dropshipping_action(params)
            elif step_type == 'wait':
                import time
                time.sleep(params.get('seconds', 1))
                return {'success': True, 'action': 'wait'}
            else:
                return {'success': False, 'error': f'Unknown step type: {step_type}'}

        except Exception as e:
            return {'success': False, 'error': str(e), 'critical_failure': True}

    def _execute_desktop_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute desktop automation action"""
        action = params.get('action')

        if action == 'type_text':
            success = self.desktop.type_text(
                params.get('text', ''),
                human_like=params.get('human_like', True)
            )
        elif action == 'move_mouse':
            success = self.desktop.move_mouse(
                params.get('x', 0),
                params.get('y', 0),
                params.get('duration', 0.5)
            )
        elif action == 'click':
            success = self.desktop.click(
                params.get('x'),
                params.get('y'),
                params.get('button', 'left')
            )
        elif action == 'open_app':
            success = self.desktop.open_application(params.get('app_name', ''))
        elif action == 'close_app':
            success = self.desktop.close_application(params.get('app_name', ''))
        elif action == 'press_key':
            success = self.desktop.press_key(
                params.get('key', ''),
                params.get('presses', 1)
            )
        elif action == 'hotkey':
            success = self.desktop.hotkey(*params.get('keys', []))
        elif action == 'take_screenshot':
            path = self.desktop.take_screenshot()
            success = path is not None
        else:
            success = False

        return {'success': success, 'action': action}

    def _execute_browser_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute browser automation action"""
        action = params.get('action')

        if action == 'navigate':
            success = self.browser.navigate(params.get('url', ''))
        elif action == 'fill_form':
            success = self.browser.fill_form(params.get('fields', {}))
        elif action == 'click':
            success = self.browser.search_and_click(params.get('selector', ''))
        elif action == 'get_text':
            text = self.browser.get_text(params.get('selector', ''))
            success = text is not None
        else:
            success = False

        return {'success': success, 'action': action}

    def _execute_booking_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute booking-related action"""
        import asyncio
        booking_request = params.get('booking_request', {})
        result = asyncio.run(self.concierge.execute_booking(booking_request))
        return {'success': result.get('success', False), 'booking_result': result}

    def _execute_vpn_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute VPN action"""
        action = params.get('action')

        if action == 'connect':
            success = self.vpn.connect(params.get('country', ''))
        elif action == 'disconnect':
            success = self.vpn.disconnect()
        elif action == 'rotate':
            success = self.vpn.rotate_location(params.get('countries', []))
        else:
            success = False

        return {'success': success, 'action': action}

    def _execute_ai_response(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI response"""
        prompt = params.get('prompt', '')
        response = self.model.generate_content(prompt)
        return {
            'success': True,
            'response': response.text,
            'action': 'ai_response'
        }

    def _execute_dropshipping_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute dropshipping automation action"""
        import asyncio
        action = params.get('action')

        try:
            if action == 'check_orders':
                orders = asyncio.run(self.dropshipping.check_orders(
                    params.get('platform', 'shopify'),
                    params.get('credentials', {})
                ))
                return {'success': True, 'orders': orders, 'action': 'check_orders'}
            elif action == 'update_inventory':
                success = asyncio.run(self.dropshipping.update_inventory(
                    params.get('platform', 'shopify'),
                    params.get('credentials', {}),
                    params.get('products', [])
                ))
                return {'success': success, 'action': 'update_inventory'}
            elif action == 'create_listing':
                success = asyncio.run(self.dropshipping.create_listing(
                    params.get('platform', 'shopify'),
                    params.get('credentials', {}),
                    params.get('product_data', {})
                ))
                return {'success': success, 'action': 'create_listing'}
            elif action == 'process_order':
                success = asyncio.run(self.dropshipping.process_order(
                    params.get('platform', 'shopify'),
                    params.get('credentials', {}),
                    params.get('order_id', '')
                ))
                return {'success': success, 'action': 'process_order'}
            else:
                return {'success': False, 'error': f'Unknown dropshipping action: {action}'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _format_response(self, execution_result: Dict[str, Any], original_request: str) -> str:
        """Format the execution result into a user-friendly response"""

        if not execution_result.get('overall_success', False):
            failed_steps = [r for r in execution_result.get('steps_executed', []) if not r.get('success', False)]
            return f"I encountered issues executing your request. Failed steps: {len(failed_steps)}. Please check the logs for details."

        steps_count = len(execution_result.get('steps_executed', []))

        # Use AI to generate a natural response
        prompt = f"""
Original request: {original_request}
Execution completed with {steps_count} steps successfully.

Generate a natural, helpful response summarizing what was accomplished.
Keep it concise but informative.
"""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except:
            return f"Request completed successfully with {steps_count} steps executed."

    def get_status(self) -> Dict[str, Any]:
        """Get assistant status"""
        return {
            'conversation_length': len(self.conversation_history),
            'tools_available': ['desktop_control', 'browser_agent', 'concierge', 'vpn_control', 'dropshipping', 'ai_response'],
            'capabilities': [
                'Desktop automation and app control',
                'Web browsing and form filling',
                'Travel booking with arbitrage',
                'Human-like typing and interaction',
                'Dropshipping business automation',
                'General AI assistance and planning'
            ]
        }

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()
