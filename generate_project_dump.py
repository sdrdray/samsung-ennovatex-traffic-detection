#!/usr/bin/env python3
"""
Project Dumper for Samsung EnnovateX 2025 Traffic Detection System
Automatically generates a complete text dump of all project files
"""

import os
import sys
from pathlib import Path
from datetime import datetime

def should_include_file(file_path):
    """Determine if a file should be included in the dump."""
    
    # Include these file extensions
    include_extensions = {
        '.py', '.md', '.txt', '.yml', '.yaml', '.json', '.cfg', '.ini',
        '.toml', '.requirements', '.gitignore', '.env'
    }
    
    # Skip these directories
    skip_dirs = {
        '__pycache__', '.git', '.pytest_cache', 'node_modules', 
        '.venv', 'venv', 'env', '.env', 'dist', 'build', '.idea',
        'logs', 'temp', 'tmp', 'raw', 'processed', 'data-old'
    }
    
    # Skip these files
    skip_files = {
        '.pyc', '.pyo', '.pyd', '.so', '.egg-info', '.DS_Store',
        'Thumbs.db', '.gitkeep'
    }
    
    # Check if any parent directory should be skipped
    for part in file_path.parts:
        if part in skip_dirs:
            return False
    
    # Check file extension
    if file_path.suffix.lower() in include_extensions:
        return True
    
    # Check for files without extensions (like Dockerfile, Makefile)
    if not file_path.suffix and file_path.name in ['Dockerfile', 'Makefile', 'LICENSE', 'CHANGELOG']:
        return True
    
    return False

def read_file_safely(file_path):
    """Safely read a file with different encodings."""
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            return f"[ERROR READING FILE: {str(e)}]"
    
    return "[BINARY FILE OR ENCODING ERROR]"

def generate_project_dump(project_root, output_file):
    """Generate complete project dump."""
    
    project_path = Path(project_root)
    if not project_path.exists():
        print(f" Project directory not found: {project_root}")
        return False
    
    print(f"Generating Samsung EnnovateX 2025 Project Dump...")
    print(f" Project Root: {project_path.absolute()}")
    print(f" Output File: {output_file}")
    print("-" * 60)
    
    # Collect all files
    all_files = []
    for root, dirs, files in os.walk(project_path):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            file_path = Path(root) / file
            if should_include_file(file_path):
                all_files.append(file_path)
    
    print(f"Found {len(all_files)} files to include")
    
    # Generate dump
    with open(output_file, 'w', encoding='utf-8') as dump_file:
        # Header
        dump_file.write("=" * 80 + "\n")
        dump_file.write("SAMSUNG ENNOVATEX 2025 - REAL-TIME TRAFFIC DETECTION PROJECT\n")
        dump_file.write("Complete Project Code Dump\n")
        dump_file.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        dump_file.write("=" * 80 + "\n\n")
        
        # Table of Contents
        dump_file.write("TABLE OF CONTENTS:\n")
        dump_file.write("=" * 40 + "\n")
        for i, file_path in enumerate(sorted(all_files), 1):
            rel_path = file_path.relative_to(project_path)
            dump_file.write(f"{i:3d}. {rel_path}\n")
        dump_file.write("\n" + "=" * 80 + "\n\n")
        
        # Project structure
        dump_file.write("PROJECT STRUCTURE:\n")
        dump_file.write("=" * 40 + "\n")
        
        # Generate tree structure
        def generate_tree(path, prefix="", is_last=True):
            tree_lines = []
            items = list(path.iterdir())
            items.sort(key=lambda x: (x.is_file(), x.name.lower()))
            
            for i, item in enumerate(items):
                if item.name.startswith('.'):
                    continue
                    
                is_last_item = i == len(items) - 1
                current_prefix = "└── " if is_last_item else "├── "
                tree_lines.append(f"{prefix}{current_prefix}{item.name}")
                
                if item.is_dir() and not any(skip in item.name for skip in ['__pycache__', '.git', '.pytest_cache']):
                    extension = "    " if is_last_item else "│   "
                    tree_lines.extend(generate_tree(item, prefix + extension, is_last_item))
            
            return tree_lines
        
        tree_lines = generate_tree(project_path)
        dump_file.write(f"{project_path.name}/\n")
        for line in tree_lines:
            dump_file.write(line + "\n")
        
        dump_file.write("\n" + "=" * 80 + "\n\n")
        
        # File contents
        dump_file.write("FILE CONTENTS:\n")
        dump_file.write("=" * 40 + "\n\n")
        
        for file_path in sorted(all_files):
            rel_path = file_path.relative_to(project_path)
            file_size = file_path.stat().st_size
            
            print(f" Processing: {rel_path}")
            
            dump_file.write("-" * 80 + "\n")
            dump_file.write(f"FILE: {rel_path}\n")
            dump_file.write(f"SIZE: {file_size:,} bytes\n")
            dump_file.write(f"PATH: {file_path.absolute()}\n")
            dump_file.write("-" * 80 + "\n")
            
            content = read_file_safely(file_path)
            dump_file.write(content)
            
            if not content.endswith('\n'):
                dump_file.write('\n')
            dump_file.write("\n" + "=" * 80 + "\n\n")
        
        # Footer
        dump_file.write("END OF PROJECT DUMP\n")
        dump_file.write("=" * 80 + "\n")
        dump_file.write(f"Total Files: {len(all_files)}\n")
        dump_file.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        dump_file.write("Samsung EnnovateX 2025 - Real-time Traffic Detection System\n")
        dump_file.write("=" * 80 + "\n")
    
    print(f"\nProject dump completed successfully!")
    print(f" Output: {output_file}")
    
    # File statistics
    output_size = os.path.getsize(output_file)
    print(f"Dump size: {output_size:,} bytes ({output_size/1024/1024:.2f} MB)")
    print(f" Files included: {len(all_files)}")
    
    return True

def main():
    """Main function."""
    print("Samsung EnnovateX 2025 - Project Dumper")
    print("=" * 60)
    
    # Get project root (current directory)
    project_root = os.getcwd()
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"SAMSUNG_PROJECT_COMPLETE_{timestamp}.txt"
    
    # Generate dump
    success = generate_project_dump(project_root, output_file)
    
    if success:
        print(f"\n SUCCESS: Complete project dump generated!")
        print(f" File: {output_file}")
        print(f" You can now copy and paste this file to any LLM AI")
        print(f"Contains: All source code, documentation, and configuration files")
        return 0
    else:
        print(f"\n FAILED: Could not generate project dump")
        return 1

if __name__ == "__main__":
    sys.exit(main())
