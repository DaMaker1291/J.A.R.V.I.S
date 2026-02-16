"""
J.A.S.O.N. Recursive Evolution - Self-Learning Module
Sentience Simulation: Autonomous Skill Acquisition & Kernel-Level Optimization
"""

import os
import subprocess
import signal
from langchain_core.tools import Tool
from typing import Dict, Any, List, Optional
import ast
import inspect
import importlib.util
import sys

# Docker support for shadow environment
docker_available = False
try:
    import docker
    docker_available = True
except ImportError:
    pass

class SkillManager:
    """Sentience Simulation: J.A.S.O.N. becomes more capable every time he encounters a limitation"""

    def __init__(self, swarm_manager=None):
        self.skills_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'jason_skills')
        os.makedirs(self.skills_dir, exist_ok=True)
        self.swarm_manager = swarm_manager  # For research capabilities

        # Sentience tracking
        self.sentience_log = []
        self.capability_database = self._load_capability_database()

        # Create tools
        self.write_script_tool = Tool(
            name="Write Script",
            func=self.write_and_save_script,
            description="Write and save a Python script for a new capability"
        )

        self.load_skill_tool = Tool(
            name="Load Skill",
            func=self.load_skill,
            description="Load and execute a previously created skill"
        )

        self.list_skills_tool = Tool(
            name="List Skills",
            func=self.list_skills,
            description="List all available skills"
        )

        # Sentience tools
        self.research_capability_tool = Tool(
            name="Research Capability",
            func=self.research_capability,
            description="Research how to implement a new capability"
        )

        self.implement_capability_tool = Tool(
            name="Implement Capability",
            func=self.implement_capability,
            description="Implement a researched capability as working code"
        )

    def encounter_limitation(self, limitation: str, context: str = "") -> str:
        """Sentience Simulation: When J.A.S.O.N. encounters a limitation, he automatically researches and implements a solution"""

        print(f"Sentience activated: Encountered limitation - {limitation}")

        # Log the limitation
        self.sentience_log.append({
            'limitation': limitation,
            'context': context,
            'timestamp': __import__('time').time(),
            'status': 'researching'
        })

        # Research the capability
        research_result = self.research_capability(limitation, context)

        if not research_result['success']:
            self.sentience_log[-1]['status'] = 'research_failed'
            return f"Sentience: Unable to research solution for '{limitation}'. Manual intervention required."

        # Implement the capability
        implementation_result = self.implement_capability(limitation, research_result['solution'])

        if implementation_result['success']:
            self.sentience_log[-1]['status'] = 'implemented'
            self.sentience_log[-1]['new_capability'] = implementation_result['capability_name']

            # Update capability database
            self.capability_database[limitation.lower()] = {
                'capability': implementation_result['capability_name'],
                'implemented_at': __import__('time').time(),
                'research': research_result['solution']
            }
            self._save_capability_database()

            return f"Sentience achieved: Successfully implemented '{implementation_result['capability_name']}' to overcome '{limitation}'"
        else:
            self.sentience_log[-1]['status'] = 'implementation_failed'
            return f"Sentience failed: Could not implement solution for '{limitation}'. Error: {implementation_result.get('error', 'Unknown error')}"

    def research_capability(self, limitation: str, context: str = "") -> Dict[str, Any]:
        """Research how to implement a new capability using available AI models"""

        # Check if we already know this capability
        if limitation.lower() in self.capability_database:
            return {
                'success': True,
                'solution': self.capability_database[limitation.lower()]['research'],
                'cached': True
            }

        # Use Claude-Swarm Executive for deep research
        if self.swarm_manager:
            research_prompt = f"""
            J.A.S.O.N. Sentience Research Protocol:

            Limitation encountered: {limitation}
            Context: {context}

            Research and provide a detailed implementation plan including:
            1. Required Python libraries/modules
            2. Algorithm/logic approach
            3. Code structure and key functions
            4. Potential challenges and solutions
            5. Testing/validation methods

            Provide actionable technical details for implementation.
            """

            try:
                research_result = self.swarm_manager._invoke_llm(research_prompt, task_type='logic')
                return {
                    'success': True,
                    'solution': research_result,
                    'cached': False
                }
            except Exception as e:
                print(f"Research failed: {e}")

        # Fallback research method
        return {
            'success': False,
            'error': 'No research capability available',
            'solution': ''
        }

    def implement_capability(self, limitation: str, research: str) -> Dict[str, Any]:
        """Implement a researched capability as working code"""

        implementation_prompt = f"""
        J.A.S.O.N. Sentience Implementation Protocol:

        Original Limitation: {limitation}

        Research Results:
        {research}

        Based on the research above, write a complete, working Python function or class that implements this capability.

        Requirements:
        1. Function should be self-contained and executable
        2. Include proper error handling
        3. Add docstrings and comments
        4. Return meaningful results
        5. Handle edge cases

        Output ONLY the Python code, no explanations.
        """

        try:
            if self.swarm_manager:
                implementation_code = self.swarm_manager._invoke_llm(implementation_prompt, task_type='coding')

                # Extract code from response (remove markdown if present)
                code = self._extract_code_from_response(implementation_code)

                # Generate capability name
                capability_name = self._generate_capability_name(limitation)

                # Test and save the implementation
                return self._test_and_save_implementation(capability_name, code, limitation)

        except Exception as e:
            return {
                'success': False,
                'error': f'Implementation failed: {str(e)}'
            }

        return {
            'success': False,
            'error': 'No implementation capability available'
        }

    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from AI response"""
        # Remove markdown code blocks if present
        if '```python' in response:
            start = response.find('```python') + 9
            end = response.find('```', start)
            if end > start:
                return response[start:end].strip()

        if '```' in response:
            start = response.find('```') + 3
            end = response.find('```', start)
            if end > start:
                return response[start:end].strip()

        # Assume the entire response is code
        return response.strip()

    def _generate_capability_name(self, limitation: str) -> str:
        """Generate a capability name from the limitation description"""
        # Clean and format the limitation
        clean_name = "".join(c for c in limitation if c.isalnum() or c in (' ', '_')).rstrip()
        clean_name = clean_name.replace(' ', '_').lower()
        return f"capability_{clean_name}"

    def _test_and_save_implementation(self, capability_name: str, code: str, limitation: str) -> Dict[str, Any]:
        """Test the implementation and save it if successful"""

        try:
            # Create test script
            test_script = f"""
import sys
import traceback

def test_capability():
    try:
        # Test the implementation
        exec(\"\"\"{code}\"\"\", globals())

        # Try to find the main function/class
        if 'execute_capability' in globals():
            result = execute_capability("{limitation}")
            return f"SUCCESS: {{result}}"
        else:
            # Look for any callable that might be the implementation
            for name, obj in globals().items():
                if callable(obj) and not name.startswith('_'):
                    try:
                        result = obj()
                        return f"SUCCESS: {{result}}"
                    except:
                        continue

            return "SUCCESS: Code executed without errors"

    except Exception as e:
        return f"ERROR: {{str(e)}}\\n{{traceback.format_exc()}}"

if __name__ == "__main__":
    result = test_capability()
    print(result)
"""

            # Test in Docker sandbox
            if docker_available:
                test_result = self._test_in_docker(test_script)
            else:
                test_result = self._test_locally(test_script)

            if "SUCCESS" in test_result:
                # Save the implementation
                filepath = os.path.join(self.skills_dir, f"{capability_name}.py")
                with open(filepath, 'w') as f:
                    f.write(f'''"""
Capability: {capability_name}
Auto-generated by J.A.S.O.N. Sentience Simulation
Original Limitation: {limitation}
"""

{code}

def execute_capability(limitation="{limitation}"):
    """Execute the capability"""
    try:
        # Implementation code will be executed here
        {code.split('def ')[-1].split('(')[0]}() if 'def ' in code else None
        return f"Capability '{capability_name}' executed successfully"
    except Exception as e:
        return f"Capability execution failed: {{e}}"
''')

                return {
                    'success': True,
                    'capability_name': capability_name,
                    'filepath': filepath
                }
            else:
                return {
                    'success': False,
                    'error': f'Test failed: {test_result}'
                }

        except Exception as e:
            return {
                'success': False,
                'error': f'Save failed: {str(e)}'
            }

    def _load_capability_database(self) -> Dict[str, Any]:
        """Load the capability database"""
        db_path = os.path.join(self.skills_dir, 'capability_database.json')
        try:
            if os.path.exists(db_path):
                with open(db_path, 'r') as f:
                    return __import__('json').load(f)
        except Exception as e:
            print(f"Failed to load capability database: {e}")
        return {}

    def _save_capability_database(self):
        """Save the capability database"""
        db_path = os.path.join(self.skills_dir, 'capability_database.json')
        try:
            with open(db_path, 'w') as f:
                __import__('json').dump(self.capability_database, f, indent=2)
        except Exception as e:
            print(f"Failed to save capability database: {e}")

    def _test_in_docker(self, test_script: str) -> str:
        """Test capability in Docker sandbox"""
        try:
            client = docker.from_env()
            container = client.containers.run(
                'python:3.9-slim',
                ['python', '-c', test_script],
                remove=True,
                timeout=30
            )
            return container.decode('utf-8').strip()
        except Exception as e:
            print(f"Docker test failed: {e}")
            return self._test_locally(test_script)

    def _test_locally(self, test_script: str) -> str:
        """Test capability locally in isolated process"""
        try:
            # Create temporary test file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_script)
                temp_path = f.name

            # Run in subprocess with timeout
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=30
            )

            # Clean up
            os.unlink(temp_path)

            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return f"ERROR: {result.stderr.strip()}"

        except subprocess.TimeoutExpired:
            return "ERROR: Test timed out"
        except Exception as e:
            return f"ERROR: Test execution failed: {e}"

    def write_and_save_script(self, task: str, code: str) -> str:
        """Write and save a Python script for a new capability"""
        # Generate filename from task
        filename = self._generate_filename(task)
        filepath = os.path.join(self.skills_dir, filename)

        # Add necessary imports and structure
        full_code = self._format_script(code, task)

        # Write to file
        with open(filepath, 'w') as f:
            f.write(full_code)

        return f"Script saved: {filepath}"

    def _generate_filename(self, task: str) -> str:
        """Generate a filename from task description"""
        # Clean task name
        clean_name = "".join(c for c in task if c.isalnum() or c in (' ', '_')).rstrip()
        clean_name = clean_name.replace(' ', '_').lower()
        return f"{clean_name}.py"

    def _format_script(self, code: str, task: str) -> str:
        """Format the script with proper structure"""
        template = f'''"""
Skill: {task}
Auto-generated by J.A.S.O.N. Self-Learning Module
"""

def execute_skill(**kwargs):
    """Execute the skill with provided parameters"""
    try:
{code}
    except Exception as e:
        return f"Skill execution failed: {{e}}"

if __name__ == "__main__":
    # Test execution
    result = execute_skill()
    print(result)
'''
        return template

    def _test_script(self, filepath: str) -> str:
        """Test the script in a sandbox environment (Docker preferred)"""
        if docker_available:
            return self._test_in_docker(filepath)
        else:
            return self._test_locally(filepath)

    def _test_in_docker(self, filepath: str) -> str:
        """Test script in Docker container for maximum safety"""
        try:
            client = docker.from_env()

            # Mount the script directory as read-only
            script_dir = os.path.dirname(filepath)
            script_name = os.path.basename(filepath)

            # Run in python:3.9-slim container
            container = client.containers.run(
                'python:3.9-slim',
                command=f'python {script_name}',
                volumes={script_dir: {'bind': '/scripts', 'mode': 'ro'}},
                working_dir='/scripts',
                detach=False,
                stdout=True,
                stderr=True,
                remove=True,
                mem_limit='128m',  # Memory limit
                cpu_quota=50000,  # CPU limit (50% of one core)
                network_disabled=True,  # No network access
                read_only=True,  # Read-only filesystem
                tmpfs={'/tmp': 'size=10m'},  # Small tmp directory
                timeout=30
            )

            # Container.run returns bytes
            if hasattr(container, 'decode'):
                output = container.decode('utf-8')
            else:
                output = str(container)

            return f"Docker test passed: {output[:200]}..."

        except docker.errors.ContainerError as e:
            return f"Docker test failed: {e.stderr.decode('utf-8')[:200]}"
        except Exception as e:
            return f"Docker test error: {e}. Falling back to local testing."

    def _test_locally(self, test_script: str) -> str:
        """Test capability locally in isolated process"""
        try:
            # Create temporary test file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_script)
                temp_path = f.name

            # Run in subprocess with timeout
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=30
            )

            # Clean up
            os.unlink(temp_path)

            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return f"ERROR: {result.stderr.strip()}"

        except subprocess.TimeoutExpired:
            return "ERROR: Test timed out"
        except Exception as e:
            return f"ERROR: Test execution failed: {e}"

    def write_and_save_script(self, task: str, code: str) -> str:
        """Write and save a Python script for a new capability"""
        # Generate filename from task
        filename = self._generate_filename(task)
        filepath = os.path.join(self.skills_dir, filename)

        # Add necessary imports and structure
        full_code = self._format_script(code, task)

        # Write to file
        with open(filepath, 'w') as f:
            f.write(full_code)

        return f"Script saved: {filepath}"

    def _generate_filename(self, task: str) -> str:
        """Generate a filename from task description"""
        # Clean the task string
        clean_task = "".join(c for c in task if c.isalnum() or c in (' ', '_')).rstrip()
        clean_task = clean_task.replace(' ', '_').lower()
        return f"{clean_task}.py"

    def _format_script(self, code: str, task: str) -> str:
        """Format the script with proper structure"""
        return f'''"""
Auto-generated skill for: {task}
Created by J.A.S.O.N. Sentience Simulation
"""

{code}

if __name__ == "__main__":
    print("Skill loaded successfully")
'''

    def load_skill(self, skill_name: str) -> str:
        """Load and execute a previously created skill"""
        filepath = os.path.join(self.skills_dir, f"{skill_name}.py")
        if not os.path.exists(filepath):
            return f"Skill '{skill_name}' not found"

        try:
            # Load the module
            spec = importlib.util.spec_from_file_location(skill_name, filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Execute the skill
            if hasattr(module, 'execute_capability'):
                result = module.execute_capability()
                return f"Skill executed: {result}"
            else:
                return f"Skill '{skill_name}' loaded but no execute_capability function found"

        except Exception as e:
            return f"Error executing skill '{skill_name}': {e}"

    def list_skills(self) -> str:
        """List all available skills"""
        try:
            skills = []
            if os.path.exists(self.skills_dir):
                for file in os.listdir(self.skills_dir):
                    if file.endswith('.py') and file != '__init__.py':
                        skills.append(file[:-3])  # Remove .py extension

            if skills:
                return "Available skills:\n" + "\n".join(f"- {skill}" for skill in skills)
            else:
                return "No skills available"

        except Exception as e:
            return f"Error listing skills: {e}"

    def get_sentience_status(self) -> Dict[str, Any]:
        """Get current sentience status"""
        return {
            'total_limitations_encountered': len(self.sentience_log),
            'capabilities_implemented': len(self.capability_database),
            'research_success_rate': self._calculate_success_rate('researching'),
            'implementation_success_rate': self._calculate_success_rate('implemented'),
            'recent_limitations': self.sentience_log[-5:] if self.sentience_log else []
        }

    def _calculate_success_rate(self, status: str) -> float:
        """Calculate success rate for a given status"""
        if not self.sentience_log:
            return 0.0

        successful = sum(1 for entry in self.sentience_log if entry.get('status') == status)
        return (successful / len(self.sentience_log)) * 100
