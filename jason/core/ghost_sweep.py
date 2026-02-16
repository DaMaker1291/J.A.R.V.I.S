"""
J.A.S.O.N. Ghost Sweep Protocol
Autonomous File Pruning and Organization System
"""

import os
import hashlib
import shutil
import gzip
import zipfile
import logging
from datetime import datetime, time
from pathlib import Path
from typing import Dict, List, Set, Optional
import threading
import time as time_module
import schedule
import json

logger = logging.getLogger(__name__)

class GhostSweepManager:
    """Ghost Sweep Protocol: Autonomous file pruning and organization"""

    def __init__(self, config: dict = None):
        self.config = config or {}

        # Default paths
        self.downloads_dir = Path.home() / "Downloads"
        self.documents_dir = Path.home() / "Documents"
        self.desktop_dir = Path.home() / "Desktop"

        # Ghost Sweep settings
        sweep_config = self.config.get('ghost_sweep', {})
        self.sweep_time = sweep_config.get('sweep_time', '03:00')  # 3 AM
        self.max_log_age_days = sweep_config.get('max_log_age_days', 30)
        self.duplicate_scan_dirs = sweep_config.get('duplicate_scan_dirs', [
            str(self.downloads_dir),
            str(self.documents_dir),
            str(self.desktop_dir)
        ])

        # File type categories
        self.file_categories = {
            'images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg'],
            'videos': ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'],
            'audio': ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma'],
            'documents': ['.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt'],
            'archives': ['.zip', '.rar', '.7z', '.tar', '.gz', '.bz2'],
            'executables': ['.exe', '.msi', '.dmg', '.pkg', '.app'],
            'code': ['.py', '.js', '.html', '.css', '.java', '.cpp', '.c', '.php']
        }

        # Statistics
        self.stats = {
            'duplicates_removed': 0,
            'logs_compressed': 0,
            'files_organized': 0,
            'space_saved_mb': 0.0,
            'last_sweep': None
        }

        # Scheduler
        self.scheduler_active = False
        self.scheduler_thread: Optional[threading.Thread] = None

        # Load previous stats
        self._load_stats()

    def start_autonomous_sweep(self) -> bool:
        """Start the autonomous sweep scheduler"""
        if self.scheduler_active:
            logger.warning("Ghost Sweep scheduler already active")
            return False

        try:
            self.scheduler_active = True
            self.scheduler_thread = threading.Thread(
                target=self._sweep_scheduler_loop,
                daemon=True
            )
            self.scheduler_thread.start()

            logger.info(f"Ghost Sweep autonomous mode activated - sweeps at {self.sweep_time}")
            return True

        except Exception as e:
            logger.error(f"Failed to start autonomous sweep: {e}")
            self.scheduler_active = False
            return False

    def stop_autonomous_sweep(self) -> bool:
        """Stop the autonomous sweep scheduler"""
        if not self.scheduler_active:
            return False

        try:
            self.scheduler_active = False
            if self.scheduler_thread:
                self.scheduler_thread.join(timeout=5)

            logger.info("Ghost Sweep autonomous mode deactivated")
            return True

        except Exception as e:
            logger.error(f"Failed to stop autonomous sweep: {e}")
            return False

    def perform_full_sweep(self) -> Dict[str, any]:
        """Perform a complete Ghost Sweep operation"""
        logger.info("Initiating full Ghost Sweep...")

        results = {
            'duplicates_removed': 0,
            'logs_compressed': 0,
            'files_organized': 0,
            'space_saved_mb': 0.0,
            'errors': []
        }

        try:
            # 1. Delete duplicate files
            dup_results = self._sweep_duplicates()
            results['duplicates_removed'] = dup_results['removed']
            results['space_saved_mb'] += dup_results['space_saved_mb']

            # 2. Compress old logs
            log_results = self._compress_old_logs()
            results['logs_compressed'] = log_results['compressed']
            results['space_saved_mb'] += log_results['space_saved_mb']

            # 3. Organize downloads and desktop
            org_results = self._organize_files()
            results['files_organized'] = org_results['organized']
            results['space_saved_mb'] += org_results['space_saved_mb']

            # Update stats
            self.stats['duplicates_removed'] += results['duplicates_removed']
            self.stats['logs_compressed'] += results['logs_compressed']
            self.stats['files_organized'] += results['files_organized']
            self.stats['space_saved_mb'] += results['space_saved_mb']
            self.stats['last_sweep'] = datetime.now().isoformat()

            self._save_stats()

            logger.info(f"Ghost Sweep completed: {results}")

        except Exception as e:
            error_msg = f"Ghost Sweep error: {e}"
            logger.error(error_msg)
            results['errors'].append(error_msg)

        return results

    def _sweep_duplicates(self) -> Dict[str, any]:
        """Find and remove duplicate files"""
        logger.info("Sweeping for duplicate files...")

        results = {
            'scanned': 0,
            'duplicates_found': 0,
            'removed': 0,
            'space_saved_mb': 0.0
        }

        file_hashes = {}  # hash -> [file_paths]

        try:
            # Scan directories for duplicates
            for dir_path in self.duplicate_scan_dirs:
                if not os.path.exists(dir_path):
                    continue

                for root, dirs, files in os.walk(dir_path):
                    # Skip hidden directories and system folders
                    dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['System', 'Library']]

                    for file in files:
                        filepath = os.path.join(root, file)

                        # Skip files that are too small or too large
                        try:
                            size = os.path.getsize(filepath)
                            if size < 1024 or size > 100 * 1024 * 1024:  # 1KB to 100MB
                                continue
                        except OSError:
                            continue

                        results['scanned'] += 1

                        # Calculate file hash
                        file_hash = self._calculate_file_hash(filepath)
                        if file_hash:
                            if file_hash not in file_hashes:
                                file_hashes[file_hash] = []
                            file_hashes[file_hash].append(filepath)

            # Process duplicates
            for hash_val, files in file_hashes.items():
                if len(files) > 1:
                    results['duplicates_found'] += len(files) - 1

                    # Keep the first file, remove the rest
                    for duplicate in files[1:]:
                        try:
                            size_mb = os.path.getsize(duplicate) / (1024 * 1024)
                            os.remove(duplicate)
                            results['removed'] += 1
                            results['space_saved_mb'] += size_mb
                            logger.info(f"Removed duplicate: {duplicate}")
                        except Exception as e:
                            logger.warning(f"Failed to remove duplicate {duplicate}: {e}")

        except Exception as e:
            logger.error(f"Error during duplicate sweep: {e}")

        return results

    def _compress_old_logs(self) -> Dict[str, any]:
        """Compress old log files"""
        logger.info("Compressing old log files...")

        results = {
            'scanned': 0,
            'compressed': 0,
            'space_saved_mb': 0.0
        }

        try:
            # Find log files
            log_dirs = [
                Path.home() / "Library" / "Logs",
                Path.home() / ".logs",
                Path.cwd() / "logs"
            ]

            cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            cutoff_date = cutoff_date.replace(day=cutoff_date.day - self.max_log_age_days)

            for log_dir in log_dirs:
                if not log_dir.exists():
                    continue

                for log_file in log_dir.rglob("*.log"):
                    if log_file.is_file():
                        results['scanned'] += 1

                        # Check file age
                        try:
                            mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                            if mtime < cutoff_date:
                                # Compress the log file
                                compressed_size = self._compress_log_file(log_file)
                                if compressed_size > 0:
                                    results['compressed'] += 1
                                    original_size = log_file.stat().st_size / (1024 * 1024)
                                    results['space_saved_mb'] += original_size - compressed_size
                                    logger.info(f"Compressed log: {log_file}")
                        except Exception as e:
                            logger.warning(f"Failed to process log {log_file}: {e}")

        except Exception as e:
            logger.error(f"Error during log compression: {e}")

        return results

    def _organize_files(self) -> Dict[str, any]:
        """Organize files in downloads and desktop"""
        logger.info("Organizing files...")

        results = {
            'scanned': 0,
            'organized': 0,
            'space_saved_mb': 0.0  # Not really space saved, but files moved
        }

        try:
            # Organize Downloads
            downloads_results = self._organize_directory(self.downloads_dir)
            results['scanned'] += downloads_results['scanned']
            results['organized'] += downloads_results['organized']

            # Organize Desktop
            desktop_results = self._organize_directory(self.desktop_dir)
            results['scanned'] += desktop_results['scanned']
            results['organized'] += desktop_results['organized']

        except Exception as e:
            logger.error(f"Error during file organization: {e}")

        return results

    def _organize_directory(self, directory: Path) -> Dict[str, any]:
        """Organize files in a specific directory"""
        results = {'scanned': 0, 'organized': 0}

        if not directory.exists():
            return results

        try:
            # Create category subdirectories
            for category in self.file_categories.keys():
                cat_dir = directory / category
                cat_dir.mkdir(exist_ok=True)

            # Scan and organize files
            for item in directory.iterdir():
                if item.is_file() and not item.name.startswith('.'):
                    results['scanned'] += 1

                    # Determine category
                    category = self._get_file_category(item)

                    if category:
                        # Move to category folder
                        target_dir = directory / category
                        target_path = target_dir / item.name

                        # Handle name conflicts
                        counter = 1
                        while target_path.exists():
                            stem = item.stem
                            suffix = item.suffix
                            target_path = target_dir / f"{stem}_{counter}{suffix}"
                            counter += 1

                        try:
                            shutil.move(str(item), str(target_path))
                            results['organized'] += 1
                            logger.info(f"Organized {item.name} -> {category}/")
                        except Exception as e:
                            logger.warning(f"Failed to organize {item}: {e}")

        except Exception as e:
            logger.error(f"Error organizing directory {directory}: {e}")

        return results

    def _get_file_category(self, filepath: Path) -> Optional[str]:
        """Determine the category of a file based on extension"""
        ext = filepath.suffix.lower()

        for category, extensions in self.file_categories.items():
            if ext in extensions:
                return category

        return None

    def _calculate_file_hash(self, filepath: str) -> Optional[str]:
        """Calculate MD5 hash of file"""
        try:
            hash_md5 = hashlib.md5()
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return None

    def _compress_log_file(self, log_file: Path) -> float:
        """Compress a log file using gzip"""
        try:
            compressed_file = log_file.with_suffix(log_file.suffix + '.gz')

            with open(log_file, 'rb') as f_in:
                with gzip.open(compressed_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Remove original file
            log_file.unlink()

            # Return compressed file size in MB
            return compressed_file.stat().st_size / (1024 * 1024)

        except Exception as e:
            logger.warning(f"Failed to compress {log_file}: {e}")
            return 0.0

    def _sweep_scheduler_loop(self):
        """Main scheduler loop for autonomous sweeps"""
        # Schedule daily sweep at specified time
        schedule.every().day.at(self.sweep_time).do(self.perform_full_sweep)

        logger.info(f"Ghost Sweep scheduler started - daily sweep at {self.sweep_time}")

        while self.scheduler_active:
            schedule.run_pending()
            time_module.sleep(60)  # Check every minute

        logger.info("Ghost Sweep scheduler stopped")

    def _load_stats(self):
        """Load previous sweep statistics"""
        stats_file = Path.home() / ".jason" / "ghost_sweep_stats.json"
        try:
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    self.stats = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load Ghost Sweep stats: {e}")

    def _save_stats(self):
        """Save sweep statistics"""
        stats_dir = Path.home() / ".jason"
        stats_dir.mkdir(exist_ok=True)
        stats_file = stats_dir / "ghost_sweep_stats.json"

        try:
            with open(stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save Ghost Sweep stats: {e}")

    def get_status(self) -> Dict[str, any]:
        """Get current Ghost Sweep status"""
        return {
            'scheduler_active': self.scheduler_active,
            'sweep_time': self.sweep_time,
            'last_sweep': self.stats.get('last_sweep'),
            'total_duplicates_removed': self.stats.get('duplicates_removed', 0),
            'total_logs_compressed': self.stats.get('logs_compressed', 0),
            'total_files_organized': self.stats.get('files_organized', 0),
            'total_space_saved_mb': self.stats.get('space_saved_mb', 0.0),
            'duplicate_scan_dirs': self.duplicate_scan_dirs
        }

    def force_sweep_now(self) -> Dict[str, any]:
        """Force an immediate sweep (for testing)"""
        logger.info("Forcing immediate Ghost Sweep...")
        return self.perform_full_sweep()
