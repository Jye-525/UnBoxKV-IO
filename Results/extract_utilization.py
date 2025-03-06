import re, sys
import csv
import math

log_file = sys.argv[1]
stats_csv_file = sys.argv[2]
fwd_csv_file = sys.argv[3]


def write_stats_csv(log_data, stats_csv_file):
    # stats_pattern = re.compile(
    # r"INFO (\d{2}-\d{2} \d{2}:\d{2}:\d{2}) metrics\.py:\d+\] "
    # r"Avg prompt throughput: (\d+\.\d+) tokens/s, "
    # r"Avg generation throughput: (\d+\.\d+) tokens/s, "
    # r"Running: (\d+) reqs, Swapped: (\d+) reqs, Pending: (\d+) reqs, "
    # r"GPU KV cache usage: (\d+\.\d+)%, CPU KV cache usage: (\d+\.\d+)%")

    stats_pattern = re.compile(
    r"INFO (\d{2}-\d{2} \d{2}:\d{2}:\d{2}) metrics\.py:\d+\] "
    r"Avg prompt throughput: (\d+\.\d+) tokens/s, "
    r"Avg generation throughput: (\d+\.\d+) tokens/s, "
    r"Running: (\d+) reqs, Swapped: (\d+) reqs, Pending: (\d+) reqs, "
    r"GPU KV cache usage: (\d+\.\d+)%, CPU KV cache usage: (\d+\.\d+)%, "
    r"Interval\(ms\): (\d+\.\d+)")

    # Write data to the CSV file with a pipe delimiter
    csv_header = ["Time elapsed (ms)", "Avg throughput (tokens/s)", "Running (reqs)", "Swapped (reqs)", "Pending (reqs)", "GPU KV cache usage (%)", "CPU KV cache usage (%)"]

    time_elapsed_ms = 0
    with open(stats_csv_file, 'w', newline='') as csvfile: 
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(csv_header)
        csvwriter.writerow([0, 0, 0, 0, 0, 0, 0])
        time_elapsed_ms = 0
        for match in stats_pattern.finditer(log_data):
            # print(match.group(2), match.group(3), match.group(4), match.group(5), match.group(6), match.group(7), match.group(8), match.group(9))
            time_elapsed_ms += float(match.group(9))
            avg_throughput = float(match.group(2)) + float(match.group(3))
            row_items = [time_elapsed_ms, avg_throughput, match.group(4), match.group(5), match.group(6), match.group(7), match.group(8)]
            csvwriter.writerow(row_items)

def write_fwd_csv(log_data, fwd_csv_file):
    #fwd_record_pattern = re.compile(
    #    r".*forward.*input_ids\.shape=torch\.Size\(\[(\d+)\]\).*"
    #    )
    #time_cost 15.7444 resumed_reqs=0, running_reqs=1 raw_running=1
    step_record_pattern = re.compile(
        r".*step.*time_cost (\d+\.\d+) resumed_reqs=(\d+), running_reqs=(\d+) raw_running=(\d+).*"
    )

    # Write data to the CSV file with a pipe delimiter
    csv_header = ["step_id", "dur(ms)", "running_reqs", "resumed_reqs", "raw_running"]

    with open(fwd_csv_file, 'w', newline='') as csvfile: 
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(csv_header)
        step_id = 0
        #for match in fwd_record_pattern.finditer(log_data):
        for match in step_record_pattern.finditer(log_data):
            #print(match.group(0))
            step_id += 1
            row_items = [match.group(1), match.group(3), match.group(2), match.group(4)]
            row_items.insert(0, step_id)
            csvwriter.writerow(row_items)


with open(log_file, 'r') as file:
    log_data = file.read()

# Read the log file and extract the matched data using Regular expression patterns
write_stats_csv(log_data, stats_csv_file)
write_fwd_csv(log_data, fwd_csv_file)
