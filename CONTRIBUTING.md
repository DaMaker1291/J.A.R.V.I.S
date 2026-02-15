# Contributing to J.A.R.V.I.S. BUT MINE

Thank you for your interest in contributing to J.A.R.V.I.S. BUT MINE! We appreciate your time and effort in making this project better.

## 🚀 Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally
   ```bash
   git clone https://github.com/your-username/J.A.R.V.I.S-BUT-MINE.git
   cd J.A.R.V.I.S-BUT-MINE
   ```
3. **Set up** the development environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e .  # Install in development mode
   ```
4. **Create a branch** for your changes
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 🔧 Development Workflow

1. **Code Style**
   - Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
   - Use type hints for better code documentation
   - Format your code with `black` before committing
     ```bash
     black .
     ```

2. **Testing**
   - Write tests for new features
   - Run tests before committing
     ```bash
     pytest
     ```

3. **Documentation**
   - Update documentation when adding new features
   - Use docstrings following Google style
   - Keep the README up to date

4. **Commit Message Guidelines**
   - Use the present tense ("Add feature" not "Added feature")
   - Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
   - Limit the first line to 72 characters or less
   - Reference issues and pull requests liberally

## 🛠 Making Changes

1. **Create a Pull Request**
   - Push your changes to your fork
   - Open a Pull Request with a clear title and description
   - Reference any related issues

2. **Code Review**
   - Be responsive to feedback
   - Keep pull requests focused and small
   - Update documentation as needed

## 📝 Reporting Issues

When reporting issues, please include:
- A clear, descriptive title
- Steps to reproduce the issue
- Expected vs. actual behavior
- Environment details (OS, Python version, etc.)
- Any relevant error messages or logs

## 📜 License

By contributing, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).

## 🙏 Thank You!

Your contributions make open source projects like this possible. Thank you for being part of our community!
