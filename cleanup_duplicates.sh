#!/bin/bash
# Script to remove duplicated files after project reorganization

echo "===================================="
echo "Cleaning up duplicated files"
echo "===================================="

# Create a backup directory for removed files
mkdir -p .duplicates_backup

# 1. Python source files that have been moved to src/ directories
echo -e "\n> Moving Python source files to backup..."
for file in $(find . -maxdepth 1 -type f -name "*.py" | grep -v "^\.\/src"); do
    # Only if it exists in the src/ directory
    filename=$(basename "$file")
    if [ -f "src/extractors/$filename" ] || [ -f "src/solvers/$filename" ] || [ -f "src/utils/$filename" ] || [ -f "src/$filename" ]; then
        echo "  - $file -> .duplicates_backup/"
        mv "$file" .duplicates_backup/
    fi
done

# 2. ASP rule files that have been moved to src/domain_rules/
echo -e "\n> Moving ASP rule files to backup..."
for file in $(find . -maxdepth 1 -type f -name "*.lp" | grep -v "^\.\/src"); do
    filename=$(basename "$file")
    if [ -f "src/domain_rules/$filename" ]; then
        echo "  - $file -> .duplicates_backup/"
        mv "$file" .duplicates_backup/
    fi
done

# 3. Test scenario data that's been moved to tests/scenarios/
echo -e "\n> Moving test data to backup..."
if [ -d "./test_data" ]; then
    echo "  - ./test_data/ -> .duplicates_backup/"
    mv ./test_data .duplicates_backup/
fi

# 4. Test scripts that have been moved to tests/scripts/
echo -e "\n> Moving test scripts to backup..."
for file in $(find . -maxdepth 1 -type f -name "test_*.sh" | grep -v "^\.\/tests"); do
    filename=$(basename "$file")
    if [ -f "tests/scripts/$filename" ]; then
        echo "  - $file -> .duplicates_backup/"
        mv "$file" .duplicates_backup/
    fi
done

# 5. Example input files that have been moved to data/example_inputs/
echo -e "\n> Moving example input files to backup..."
for file in $(find . -maxdepth 1 -type f -name "*.txt" | grep -v "^\.\/data" | grep -v "requirements.txt"); do
    filename=$(basename "$file")
    if [ -f "data/example_inputs/$filename" ]; then
        echo "  - $file -> .duplicates_backup/"
        mv "$file" .duplicates_backup/
    fi
done

# 6. Clean up any remaining temporary files
echo -e "\n> Cleaning up temporary files..."
if [ -f "extracted_facts.lp" ]; then
    echo "  - ./extracted_facts.lp -> .duplicates_backup/"
    mv extracted_facts.lp .duplicates_backup/
fi

echo -e "\n===================================="
echo "Cleanup complete!"
echo "All duplicated files have been moved to .duplicates_backup/"
echo "====================================" 