"""
C21040310 - Jake Palmer
Gas and Water Pipeline Analysis using Machine Learning Techniques
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# Define attack type descriptions
attack_types = {
    0: 'Normal - Not part of an attack',
    1: 'NMRI - Naive Malicious Response Injection Attack',
    2: 'CMRI - Complex Malicious Response Injection Attack',
    3: 'MSCI - Malicious State Command Injection Attack',
    4: 'MPCI - Malicious Parameter Command Injection Attack',
    5: 'MFCI - Malicious Function Command Injection Attack',
    6: 'DoS - Denial of Service',
    7: 'Reconnaissance - Probe for System Information'
}

# Define custom color map for attack types
attack_colors = {
    0: 'blue',     # Normal
    1: 'green',    # NMRI
    2: 'red',      # CMRI
    3: 'orange',   # MSCI
    4: 'purple',   # MPCI
    5: 'yellow',   # MFCI
    6: 'cyan',     # DoS
    7: 'magenta'   # Reconnaissance
}

# Loading the dataset
def load_data(file_path, column_names, skiprows):
    df = pd.read_csv(file_path, encoding='utf-8', names=column_names, header=None, skiprows=skiprows)
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    
    return df

# Standardising the Features
def preprocess_data(df, features):
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df

# Detecting anomalies using Isolation Forest and Local Outlier Detection
def detect_anomalies(df, features):
    isolation_forest = IsolationForest(contamination=0.1, random_state=30)
    df['anomaly_isolation_forest'] = isolation_forest.fit_predict(df[features])

    lof = LocalOutlierFactor(n_neighbors=50, contamination=0.1)
    df['anomaly_local_outlier_factor'] = lof.fit_predict(df[features])

    df['anomaly_combined'] = df['anomaly_isolation_forest'] + df['anomaly_local_outlier_factor'] # Combines the anomaly scores
    
    return df

def adjust_anomaly_scores(df, features):
    rf_regressor = RandomForestRegressor()
    X = df[features]
    y = df['anomaly_combined']
    rf_regressor.fit(X, y)
    df['anomaly_rf_adjusted'] = rf_regressor.predict(X)
    return df

# Plotting Anomalies
def visualize_anomalies(df, features):
    figure, subplot_array = plt.subplots(len(features), 1, figsize=(10, 5 * len(features)), sharex=True)
    for i, feature in enumerate(features):
        colors = df['result'].map(lambda x: attack_colors[x])  # Color code based on attack type
        subplot_array[i].scatter(df['time'], df[feature], c=colors, alpha=0.5)
        subplot_array[i].set_title(f'Anomaly Detection for {feature}')
        subplot_array[i].set_ylabel(feature)
        subplot_array[i].set_xlabel('Time')
        subplot_array[i].grid(True)
        subplot_array[i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
        subplot_array[i].set_aspect('auto', adjustable='box')  # Adjust aspect ratio to fit the data

    # Create legend
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=attack_types[key], markerfacecolor=color, markersize=10) for key, color in attack_colors.items()]
    plt.subplots_adjust(hspace=0.5)
    plt.legend(handles=legend_handles, loc='upper right')
    plt.tight_layout(pad=5.0)
    plt.show()

def estimate_result(df, features):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df[df['anomaly_combined'] == 0][features], 
                                                        df[df['anomaly_combined'] == 0]['result'], 
                                                        test_size=0.2, random_state=42)
    
    classifiers = { # More can be added if needed.
        "Random Forest": RandomForestClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Support Vector Machine": SVC(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Gaussian Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression()
    }
    
    results = {}
    
    for name, classifier in classifiers.items():
        # Train the classifier
        classifier.fit(X_train, y_train)
        
        # Predict 'result' column on the entire dataset
        df[f'estimated_result_{name}'] = classifier.predict(df[features])
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(df[df['anomaly_combined'] == 0]['result'], df[df['anomaly_combined'] == 0][f'estimated_result_{name}'])
        precision = precision_score(df[df['anomaly_combined'] == 0]['result'], df[df['anomaly_combined'] == 0][f'estimated_result_{name}'], average='weighted')
        recall = recall_score(df[df['anomaly_combined'] == 0]['result'], df[df['anomaly_combined'] == 0][f'estimated_result_{name}'], average='weighted')
        f1 = f1_score(df[df['anomaly_combined'] == 0]['result'], df[df['anomaly_combined'] == 0][f'estimated_result_{name}'], average='weighted')
        
        results[name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
    
    return df, results

def evaluate_result_estimation(df):
    # Evaluate accuracy of 'estimated_result' compared to actual 'result'
    accuracy = accuracy_score(df[df['anomaly_combined'] == 0]['result'], df[df['anomaly_combined'] == 0]['estimated_result'])
    return accuracy

def main():
    # Load the dataset
    file_path = 'Gas_Pipeline_Raw.csv'
    column_names = [
        'command_address', 'response_address', 'command_memory', 'response_memory', 
        'command_memory_count', 'response_memory_count', 'comm_read_function', 
        'comm_write_fun', 'resp_read_fun', 'resp_write_fun', 'sub_function', 
        'command_length', 'resp_length', 'gain', 'reset', 'deadband', 'cycletime', 
        'rate', 'setpoint', 'control_mode', 'control_scheme', 'pump', 'solenoid', 
        'crc_rate', 'measurement', 'time', 'result'
    ]
    
    print("Processing Gas Pipeline Data...")
    df = load_data(file_path, column_names, skiprows=32)

    # Selecting only relevant features for anomaly detection
    features = ['command_memory', 'response_address', 'command_length', 'pump', 'solenoid', 
        'crc_rate', 'measurement']

    # Preprocess the data
    df = preprocess_data(df, features)

    # Detect anomalies
    df = detect_anomalies(df, features)

    # Adjust anomaly scores
    df = adjust_anomaly_scores(df, features)

    # Estimate the 'result' column based on non-anomalous instances
    df, results = estimate_result(df, features)

    # Visualize anomalies
    visualize_anomalies(df, features)

    # Print the results of prediction methods for the gas pipeline dataset
    for name, result in results.items():
        print(f"{name} - Accuracy: {result['accuracy']}, Precision: {result['precision']}, Recall: {result['recall']}, F1-score: {result['f1_score']}")

    # Write anomalous data along with true result and predicted result to a CSV file
    anomalous_data = df[df['anomaly_combined'] != 0]
    anomalous_data.to_csv('predicted_data_gas_pipeline.csv', index=False)

    # For the water pipeline dataset
    file_path_water = 'Water_Pipeline_Raw.csv'
    column_names_water = [
        'command_address', 'response_address', 'command_memory', 'response_memory', 
        'command_memory_count', 'response_memory_count', 'comm_read_function', 
        'comm_write_fun', 'resp_read_fun', 'resp_write_fun', 'sub_function', 
        'command_length', 'resp_length', 'HH', 'H', 'L', 'LL', 'control_mode', 
        'control_scheme', 'pump', 'crc_rate', 'measurement', 'time', 'result'
    ]
    
    print("Processing Water Pipeline Data...")
    df_water = load_data(file_path_water, column_names_water, skiprows=29)

    # Selecting only relevant features for anomaly detection
    features_water = ['command_memory', 'response_address', 'command_length', 'pump', 'crc_rate', 'measurement', 'time']

    # Preprocess the data
    df_water = preprocess_data(df_water, features_water)

    # Detect anomalies
    df_water = detect_anomalies(df_water, features_water)

    # Adjust anomaly scores
    df_water = adjust_anomaly_scores(df_water, features_water)

    # Estimate the 'result' column based on non-anomalous instances
    df_water, results_water = estimate_result(df_water, features_water)
    
    # Visualize anomalies
    visualize_anomalies(df_water, features_water)

    # Write anomalous data along with true result and predicted result to a CSV file
    anomalous_data_water = df_water[df_water['anomaly_combined'] != 0]
    anomalous_data_water.to_csv('predicted_data_water_pipeline.csv', index=False)

    # Print the results of prediction methods for water pipeline dataset
    for name, result in results_water.items():
        print(f"{name} - Accuracy: {result['accuracy']}, Precision: {result['precision']}, Recall: {result['recall']}, F1-score: {result['f1_score']}")

if __name__ == "__main__":
    main()