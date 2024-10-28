import os
import glob

def cleanup_test_files(root_dir, patterns_to_keep=None, dry_run=True):
    """
    Recursively walk through directories and remove all .c files except those matching specified patterns
    
    Args:
        root_dir (str): Root directory to start the search
        patterns_to_keep (list): List of file endings to keep (e.g., ['01.c', '02.c', '03.c'])
        dry_run (bool): If True, only prints what would be done without actually deleting
    """
    if patterns_to_keep is None:
        patterns_to_keep = ['01.c']
    
    deleted_count = 0
    kept_count = 0
    kept_files_by_pattern = {pattern: 0 for pattern in patterns_to_keep}
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Get all .c files in current directory
        c_files = glob.glob(os.path.join(dirpath, "*.c"))
        
        for c_file in c_files:
            # Check if file matches any of the patterns to keep
            should_keep = any(c_file.endswith(pattern) for pattern in patterns_to_keep)
            
            if should_keep:
                if dry_run:
                    print(f"Would keep: {c_file}")
                # Count files kept by pattern
                for pattern in patterns_to_keep:
                    if c_file.endswith(pattern):
                        kept_files_by_pattern[pattern] += 1
                kept_count += 1
            else:
                if dry_run:
                    print(f"Would delete: {c_file}")
                else:
                    try:
                        os.remove(c_file)
                        print(f"Deleted: {c_file}")
                    except Exception as e:
                        print(f"Error deleting {c_file}: {e}")
                deleted_count += 1
    
    print(f"\nSummary:")
    print(f"Total files kept: {kept_count}")
    for pattern, count in kept_files_by_pattern.items():
        print(f"  Files ending with {pattern}: {count}")
    print(f"Files {'marked for deletion' if dry_run else 'deleted'}: {deleted_count}")

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean up JULIET test suite files.')
    parser.add_argument('directory', help='Directory to process')
    parser.add_argument('--patterns', nargs='+', default=['01.c'],
                      help='File patterns to keep (e.g., 01.c 02.c 03.c)')
    parser.add_argument('--execute', action='store_true',
                      help='Execute deletion (without this flag, runs in dry-run mode)')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory")
        sys.exit(1)
    
    print("Patterns to keep:", args.patterns)
    print("WARNING: This script will delete files. Please make sure you have a backup.")
    
    if args.execute:
        confirm = input("Are you sure you want to proceed with deletion? (yes/no): ")
        if confirm.lower() != "yes":
            print("Operation cancelled.")
            sys.exit(0)
    else:
        print("Running in dry-run mode. Use --execute to actually delete files.")
    
    cleanup_test_files(args.directory, patterns_to_keep=args.patterns, dry_run=not args.execute)