// J.A.S.O.N. Professional Desktop Automation UI Script

// Global state for demo
let commandHistory = [
    "Welcome to J.A.S.O.N. Professional Desktop Automation 🤖",
    "Type a command below to see J.A.S.O.N. in action!"
];

// Mock system data for demo
let mockSystemData = {
    cpu: Math.floor(Math.random() * 100),
    memory: Math.floor(Math.random() * 100),
    disk: Math.floor(Math.random() * 100),
    processes: Math.floor(Math.random() * 500) + 100
};

// Update system data periodically for demo
setInterval(() => {
    mockSystemData = {
        cpu: Math.max(0, Math.min(100, mockSystemData.cpu + (Math.random() - 0.5) * 10)),
        memory: Math.max(0, Math.min(100, mockSystemData.memory + (Math.random() - 0.5) * 5)),
        disk: Math.max(0, Math.min(100, mockSystemData.disk + (Math.random() - 0.5) * 2)),
        processes: Math.max(50, Math.min(1000, mockSystemData.processes + (Math.random() - 0.5) * 20))
    };
    updateSystemDisplay();
}, 2000);

// Initialize the UI
document.addEventListener('DOMContentLoaded', function() {
    updateCommandHistory();
    updateSystemDisplay();

    // Add smooth scrolling to navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
});

// Handle command execution
function handleCommand(event) {
    if (event.key === 'Enter') {
        executeCommand();
    }
}

function executeCommand() {
    const input = document.getElementById('commandInput');
    const command = input.value.trim();

    if (command) {
        // Add command to history
        addToCommandHistory(`J.A.S.O.N. > ${command}`);

        // Process the command
        processCommand(command);

        // Clear input
        input.value = '';
    }
}

function processCommand(command) {
    const lowerCommand = command.toLowerCase();

    if (lowerCommand.includes('system status') || lowerCommand.includes('status')) {
        addToCommandHistory("📊 Retrieving system status...");
        setTimeout(() => {
            addToCommandHistory("✅ System status retrieved successfully!");
            addToCommandHistory(`   CPU Usage: ${mockSystemData.cpu.toFixed(1)}%`);
            addToCommandHistory(`   Memory: ${mockSystemData.memory.toFixed(1)}% used`);
            addToCommandHistory(`   Disk: ${mockSystemData.disk.toFixed(1)}% used`);
            addToCommandHistory(`   Active Processes: ${Math.floor(mockSystemData.processes)}`);
        }, 500);
    }
    else if (lowerCommand.includes('list processes') || lowerCommand.includes('processes')) {
        addToCommandHistory("📋 Retrieving process list...");
        setTimeout(() => {
            addToCommandHistory("✅ Process list retrieved!");
            addToCommandHistory("   Top processes by CPU usage:");
            const processes = [
                { name: "Google Chrome", cpu: (Math.random() * 20).toFixed(1), pid: Math.floor(Math.random() * 10000) },
                { name: "Terminal", cpu: (Math.random() * 15).toFixed(1), pid: Math.floor(Math.random() * 10000) },
                { name: "Visual Studio Code", cpu: (Math.random() * 10).toFixed(1), pid: Math.floor(Math.random() * 10000) },
                { name: "Finder", cpu: (Math.random() * 5).toFixed(1), pid: Math.floor(Math.random() * 10000) },
                { name: "Slack", cpu: (Math.random() * 8).toFixed(1), pid: Math.floor(Math.random() * 10000) }
            ];
            processes.forEach(proc => {
                addToCommandHistory(`   ${proc.name} (PID: ${proc.pid}) - CPU: ${proc.cpu}%`);
            });
        }, 800);
    }
    else if (lowerCommand.includes('arrange windows') || lowerCommand.includes('windows')) {
        addToCommandHistory("🪟 Arranging windows in grid layout...");
        setTimeout(() => {
            addToCommandHistory("✅ Windows arranged successfully!");
            addToCommandHistory("   📱 Applied 2x2 grid layout to all open windows");
            addToCommandHistory("   📱 Windows positioned for optimal productivity");
        }, 600);
    }
    else if (lowerCommand.includes('productivity mode') || lowerCommand.includes('productivity')) {
        addToCommandHistory("🚀 Activating productivity mode...");
        setTimeout(() => {
            addToCommandHistory("✅ Productivity mode activated!");
            addToCommandHistory("   📵 Closed distracting applications");
            addToCommandHistory("   📱 Launched productivity apps");
            addToCommandHistory("   🪟 Arranged windows optimally");
            addToCommandHistory("   ⚡ System optimizations applied");
        }, 1000);
    }
    else if (lowerCommand.includes('security scan') || lowerCommand.includes('security')) {
        addToCommandHistory("🔒 Running security scan...");
        setTimeout(() => {
            addToCommandHistory("✅ Security scan completed!");
            addToCommandHistory("   🛡️ Checked for suspicious processes");
            addToCommandHistory("   🌐 Monitored network activity");
            addToCommandHistory("   ✅ No security threats detected");
            addToCommandHistory("   📊 Scan completed in 2.3 seconds");
        }, 1200);
    }
    else if (lowerCommand.includes('kill process') || lowerCommand.includes('terminate')) {
        const pidMatch = command.match(/(\d+)/);
        if (pidMatch) {
            const pid = pidMatch[1];
            addToCommandHistory(`⚠️ Terminating process ${pid}...`);
            setTimeout(() => {
                addToCommandHistory(`✅ Process ${pid} terminated successfully!`);
            }, 300);
        } else {
            addToCommandHistory("❌ Please specify a process ID (PID) to terminate");
            addToCommandHistory("   Example: kill process 1234");
        }
    }
    else if (lowerCommand.includes('help') || lowerCommand.includes('?')) {
        addToCommandHistory("📚 Available commands:");
        addToCommandHistory("   • system status - View system information");
        addToCommandHistory("   • list processes - Show running processes");
        addToCommandHistory("   • arrange windows - Organize windows in grid");
        addToCommandHistory("   • productivity mode - Activate productivity workflow");
        addToCommandHistory("   • security scan - Run security check");
        addToCommandHistory("   • kill process <PID> - Terminate specific process");
        addToCommandHistory("   • help - Show this help message");
    }
    else {
        addToCommandHistory(`❓ Unknown command: "${command}"`);
        addToCommandHistory("   Type 'help' for available commands");
    }
}

function addToCommandHistory(text) {
    commandHistory.push(text);
    if (commandHistory.length > 20) {
        commandHistory = commandHistory.slice(-20);
    }
    updateCommandHistory();
}

function updateCommandHistory() {
    const historyElement = document.getElementById('commandHistory');
    historyElement.innerHTML = commandHistory.map(line =>
        `<div class="command-line">${line}</div>`
    ).join('');
    historyElement.scrollTop = historyElement.scrollHeight;
}

function updateSystemDisplay() {
    document.getElementById('cpuUsage').textContent = `${mockSystemData.cpu.toFixed(1)}%`;
    document.getElementById('memoryUsage').textContent = `${mockSystemData.memory.toFixed(1)}%`;
    document.getElementById('diskUsage').textContent = `${mockSystemData.disk.toFixed(1)}%`;
    document.getElementById('processCount').textContent = Math.floor(mockSystemData.processes);
}

// Demo functions
function startDemo() {
    const commands = [
        "system status",
        "list processes",
        "arrange windows",
        "productivity mode",
        "security scan"
    ];

    let index = 0;
    const demoInterval = setInterval(() => {
        if (index < commands.length) {
            document.getElementById('commandInput').value = commands[index];
            executeCommand();
            index++;
        } else {
            clearInterval(demoInterval);
            setTimeout(() => {
                addToCommandHistory("🎉 Demo completed! Try your own commands above.");
            }, 1000);
        }
    }, 2000);

    addToCommandHistory("🚀 Starting automated demo...");
}

function showSystemStatus() {
    // Scroll to demo section
    document.getElementById('demo').scrollIntoView({
        behavior: 'smooth',
        block: 'start'
    });

    // Update system display immediately
    updateSystemDisplay();

    // Add some visual feedback
    addToCommandHistory("📊 System status updated in real-time!");
}

// Add some interactive effects
document.addEventListener('DOMContentLoaded', function() {
    // Add hover effects to feature cards
    document.querySelectorAll('.feature-card').forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-10px) scale(1.02)';
        });

        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
        });
    });

    // Add typing effect to command input focus
    const commandInput = document.getElementById('commandInput');
    commandInput.addEventListener('focus', function() {
        this.parentElement.style.boxShadow = '0 0 20px rgba(102, 126, 234, 0.3)';
    });

    commandInput.addEventListener('blur', function() {
        this.parentElement.style.boxShadow = 'none';
    });
});

// Performance monitoring for demo
let fpsCounter = 0;
let lastTime = Date.now();

function updateFPS() {
    const currentTime = Date.now();
    fpsCounter++;

    if (currentTime - lastTime >= 1000) {
        // Update FPS display if we had an element for it
        lastTime = currentTime;
        fpsCounter = 0;
    }

    requestAnimationFrame(updateFPS);
}

updateFPS();
