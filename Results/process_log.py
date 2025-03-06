import re, sys

input_file = sys.argv[1]
cmd_type = sys.argv[2]

file_name=input_file

# open the file and read content
with open(file_name, 'r') as file:
    log_data = file.read()

# filter out the matched data using Regular expression patterns
# extract "First token time"
if cmd_type == "gen_to_file" :
    
    pattern_reqs_greater_than_1 = re.compile(r".*(\d{2}:\d{2}:\d{2}).*take (\d+\.\d+) seconds.*parallel_reqs = (\d+).*parallel_ids=(\[.*?\])")
    filtered_reqs_greater_than_1 = []
    last_currency = None
    last_parallel_ids = None
    last_time_string = None

    for match in pattern_reqs_greater_than_1.finditer(log_data):
        # print(f"matched item: {match.group(0)}")
        time_string = match.group(1)
        time_cost = float(match.group(2))
        cur_currency = int(match.group(3))
        cur_parallel_ids = match.group(4)
        if cur_currency > 1:
            if last_currency != cur_currency:
                filtered_reqs_greater_than_1.append(match.group(0))
            elif last_currency == cur_currency and last_parallel_ids != cur_parallel_ids:
                # This condition have problems for prefill chunks, calculation about the overlapping time is not correct.
                filtered_reqs_greater_than_1.append(match.group(0))
            elif last_currency == cur_currency and last_parallel_ids == cur_parallel_ids and last_time_string != time_string:
                filtered_reqs_greater_than_1.append(match.group(0))
            
            last_currency = cur_currency
            last_parallel_ids = cur_parallel_ids
            last_time_string = time_string
        else:
            filtered_reqs_greater_than_1.append(match.group(0))
            last_currency = None
            last_parallel_ids = None

    # Combine the results
    all_filtered_entries = filtered_reqs_greater_than_1
    # Write the filtered entries to an output file
    out_file_name="filtered_log.txt"
    with open(out_file_name, 'w') as output_file:
        for entry in all_filtered_entries:
            output_file.write(entry + '\n')

    print("Filtered log entries have been written to ./filtered_log.txt")

elif cmd_type == "sum_gen" or cmd_type == "sum_swap_out" or cmd_type == "sum_swap_in":
    total = 0.0
    pattern = re.compile(r"\d+\.\d+")

    # Find all matches of the pattern in the log_data
    matches = pattern.findall(log_data)

    # Convert the matches to floats and calculate the sum
    total = sum(float(match) for match in matches)

    # Print the total
    print(f'The sum of the extracted numbers from {len(matches)} items is: {total}')

else:
    print("Invalid command type. Please provide a valid command type: gen_to_file, sum_gen, sum_swap_out, or sum_swap_in.")
