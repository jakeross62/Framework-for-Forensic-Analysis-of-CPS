"""
C21040310 - Jake Palmer
Attack Category Correlation to Correlate Digital and Physical Consequences
"""
import pandas as pd

def calculate_differences(normal_readings, anomalous_readings):
    # Align the columns of normal and anomalous readings
    common_columns = normal_readings.columns.intersection(anomalous_readings.columns)
    normal_aligned = normal_readings[common_columns]
    anomalous_aligned = anomalous_readings[common_columns]
    
    # Calculate the differences
    differences = anomalous_aligned.sub(normal_aligned.mean())
    return differences

def get_column_names_diff(differences, normal_readings):
    normal_mean = normal_readings.mean()
    return differences.columns[differences.abs().mean() > (normal_mean * 0.1)]  # Adjust the threshold as needed

def filter_by_attack_type(df, attack_type):
    return df[df['result'] == attack_type]

def save_differences_to_txt(output_data, pipeline_name):
    filename = f'differences_{pipeline_name}.txt'
    with open(filename, 'w') as txtfile:
        for line in output_data:
            txtfile.write('\n'.join(map(str, line)) + '\n')

def main():
    pipelines = {
        'water': {
            'file_path': 'Water_Pipeline_Raw.csv',
            'skiprows': 29,
            'column_names': [
                'command_address', 'response_address', 'command_memory', 'response_memory', 
                'command_memory_count', 'response_memory_count', 'comm_read_function', 
                'comm_write_fun', 'resp_read_fun', 'resp_write_fun', 'sub_function', 
                'command_length', 'resp_length', 'HH', 'H', 'L', 'LL', 'control_mode', 
                'control_scheme', 'pump', 'crc_rate', 'measurement', 'time', 'result'
            ]
        },
        'gas': {
            'file_path': 'Gas_Pipeline_Raw.csv',
            'skiprows': 32,
            'column_names': [
                'command_address', 'response_address', 'command_memory', 'response_memory', 
                'command_memory_count', 'response_memory_count', 'comm_read_function', 
                'comm_write_fun', 'resp_read_fun', 'resp_write_fun', 'sub_function', 
                'command_length', 'resp_length', 'gain', 'reset', 'deadband', 'cycletime', 
                'rate', 'setpoint', 'control_mode', 'control_scheme', 'pump', 'solenoid', 
                'crc_rate', 'measurement', 'time', 'result'
            ]
        }
    }

    attack_types = {
        1: "NMRI - Naive Malicious Response Injection Attack",
        2: "CMRI - Complex Malicious Response Injection Attack",
        3: "MSCI - Malicious State Command Injection Attack",
        4: "MPCI - Malicious Parameter Command Injection Attack",
        5: "MFCI - Malicious Function Command Injection Attack",
        6: "DoS - Denial of Service",
        7: "Reconnaissance - Probe for System Information"
    }

    for pipeline_name, pipeline_info in pipelines.items():
        df = pd.read_csv(pipeline_info['file_path'], skiprows=pipeline_info['skiprows'], names=pipeline_info['column_names'])
        df.drop(columns=['command_address', 'response_address', 'time'], inplace=True)
        output_data = []

        for attack_type, attack_description in attack_types.items():
            if attack_type != 0:  # Exclude attack type 0 as this is normal data for each pipeline
                # Filter out rows with the current attack type
                normal_readings = filter_by_attack_type(df, 0)
                anomalous_readings = filter_by_attack_type(df, attack_type)
                differences = calculate_differences(normal_readings, anomalous_readings)
                column_names_diff = get_column_names_diff(differences, normal_readings)

                # Output the data
                output_data.append([f"\nAttack Type: {attack_type} - {attack_description}"])
                output_data.append([f"Columns with significant differences in {pipeline_name} pipeline:"])
                output_data.append(column_names_diff)

        # Save the differences to a single TXT file
        save_differences_to_txt(output_data, pipeline_name)

if __name__ == "__main__":
    main()
