#!/bin/bash
Type=$1 # recompute=0 swap=1
log_file=$2
echo "log_file="${log_file}

cur_path=`pwd`
log_file="$cur_path/$log_file"

if [ $Type -eq 0 ]; then
    # Recompute
    # Extract all the recompute
    sed -n '/Start benchmarking/,$p' $log_file | grep "First" > temp_1.txt
    cat temp_1.txt
    python3 ./process_log.py temp_1.txt "gen_to_file"
    cat filtered_log.txt

    # Extract all the recompute cost time and sum all of them
    python3 ./process_log.py filtered_log.txt "sum_gen"

     rm -f temp_1.txt
     rm -f filtered_log.txt

else
    # Swap
    # Extract all the gen_t
    sed -n '/Start benchmarking/,$p' $log_file | grep "First" > temp_1.txt
    cat temp_1.txt
    python3 ./process_log.py temp_1.txt "gen_to_file"
    cat filtered_log.txt

    # read temp.txt to get the sum of total gen_t
    echo "Get sum gen_t result: "
    python3 ./process_log.py filtered_log.txt "sum_gen"
    
    rm -f temp_1.txt
    rm -f filerted_log.txt

    # Extract all the swap_in
    sed -n '/Start benchmarking/,$p' $log_file | grep "swap in" > temp_1.txt
    
    cat temp_1.txt

    # read temp.txt to get the sum of total swap_in
    echo "get sum swap_in result: "
    python3 ./process_log.py temp_1.txt "sum_swap_in"
    rm -f temp_1.txt

    # Extract all the swap_out
    sed -n '/Start benchmarking/,$p' $log_file | grep "swap out" > temp_1.txt
    cat temp_1.txt
    # Initialize a variable to hold the sum
    echo "get sum swap_out result: "
    python3 ./process_log.py temp_1.txt "sum_swap_out"
    rm -f temp_1.txt

fi
# Extract all the swap
