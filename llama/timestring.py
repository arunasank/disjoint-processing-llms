from datetime import datetime, timedelta

def add_times(time_list):
    total_time = timedelta()

    # Iterate over the list of time strings
    for time_str in time_list:
        # Parse time string into a timedelta object
        time_delta = datetime.strptime(time_str, '%H:%M:%S') - datetime.strptime('00:00:00', '%H:%M:%S')
        # Add the timedelta to the total time
        total_time += time_delta

    # Convert total time back to 'HH:MM:SS' format
    total_time_str = str(total_time)
    
    return total_time_str

# Example usage
time_list = ['00:21:33', '00:24:15', '00:24:25', '00:24:31', '00:20:34', '00:23:09', '00:22:54', '00:19:45', '00:23:23', '00:23:01']

result = add_times(time_list)
print("Result:", result)

