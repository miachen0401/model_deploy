"""
Pre-deployment validation script.
Checks that all necessary files and configurations are in place.
"""
import os
import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_file_exists(file_path: str, required: bool = True) -> bool:
    """Check if a file exists"""
    exists = Path(file_path).exists()
    status = "✓" if exists else "✗"

    if exists:
        logger.info(f"{status} Found: {file_path}")
        return True
    else:
        level = logging.ERROR if required else logging.WARNING
        logger.log(level, f"{status} Missing: {file_path}")
        return False


def check_env_variables() -> bool:
    """Check required environment variables"""
    logger.info("\nChecking environment variables...")

    from dotenv import load_dotenv
    load_dotenv()

    required_vars = ['HF_TOKEN_READ']
    all_present = True

    for var in required_vars:
        value = os.getenv(var)
        if value:
            logger.info(f"✓ {var} is set")
        else:
            logger.error(f"✗ {var} is NOT set")
            all_present = False

    return all_present


def check_config_file() -> bool:
    """Check config.yml structure"""
    logger.info("\nChecking config.yml...")

    try:
        import yaml
        with open('config.yml', 'r') as f:
            config = yaml.safe_load(f)

        if 'HUGGINGFACE_MODEL' in config and 'NAME' in config['HUGGINGFACE_MODEL']:
            model_name = config['HUGGINGFACE_MODEL']['NAME']
            logger.info(f"✓ Model name configured: {model_name}")
            return True
        else:
            logger.error("✗ config.yml missing HUGGINGFACE_MODEL.NAME")
            return False

    except Exception as e:
        logger.error(f"✗ Error reading config.yml: {str(e)}")
        return False


def check_dependencies() -> bool:
    """Check if required packages are installed"""
    logger.info("\nChecking Python dependencies...")

    required_packages = [
        'fastapi',
        'uvicorn',
        'transformers',
        'torch',
        'yaml',
        'dotenv',
        'pydantic'
    ]

    all_installed = True

    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} installed")
        except ImportError:
            logger.error(f"✗ {package} NOT installed")
            all_installed = False

    return all_installed


def validate_all() -> bool:
    """Run all validation checks"""
    logger.info("=" * 60)
    logger.info("Validating Deployment Setup")
    logger.info("=" * 60)

    checks = []

    # Check required files
    logger.info("\nChecking required files...")
    checks.append(check_file_exists('app.py'))
    checks.append(check_file_exists('requirements.txt'))
    checks.append(check_file_exists('config.yml'))
    checks.append(check_file_exists('render.yaml'))
    checks.append(check_file_exists('.env'))
    checks.append(check_file_exists('.gitignore'))

    # Check optional but recommended files
    logger.info("\nChecking optional files...")
    check_file_exists('test_api.py', required=False)
    check_file_exists('example_usage.py', required=False)
    check_file_exists('README.md', required=False)

    # Check environment variables
    checks.append(check_env_variables())

    # Check config file
    checks.append(check_config_file())

    # Check dependencies (optional for deployment, required for local)
    logger.info("\nChecking dependencies (required for local testing)...")
    deps_installed = check_dependencies()
    if not deps_installed:
        logger.warning("! Dependencies not installed. Run: pip install -r requirements.txt")

    # Summary
    logger.info("\n" + "=" * 60)
    if all(checks):
        logger.info("✓ All required checks PASSED!")
        logger.info("You're ready to deploy to Render.")
        logger.info("\nNext steps:")
        logger.info("1. Push code to GitHub")
        logger.info("2. Create Web Service on Render")
        logger.info("3. Add HF_TOKEN_READ to Render environment variables")
        logger.info("4. Deploy!")
        return True
    else:
        logger.error("✗ Some checks FAILED. Please fix the issues above.")
        return False


if __name__ == "__main__":
    success = validate_all()
    sys.exit(0 if success else 1)
