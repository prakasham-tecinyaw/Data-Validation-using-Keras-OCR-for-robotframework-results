# src/logger.py
def log_results(results, log_file):
    with open(log_file, 'w') as f:  # Open in write mode to overwrite
        for result in results:
            f.write(result + '\n')  # Write each result on a new line