import pandas as pd
import os

def reorder_labels(input_file, output_file):
    """
    Reorders columns in the labels CSV file to match the desired order.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to save the reordered CSV file
    """
    print(f"Reading labels from {input_file}...")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Define the new column order
    # Original: filename,fog,glare,road,traffic,weather,scene,timeofday
    # Desired: filename,timeofday,glare,weather,fog,road,traffic,scene
    reordered_columns = ['filename', 'timeofday', 'glare', 'weather', 'fog', 'road', 'traffic', 'scene']
    df_reordered = df[reordered_columns]
    
    # Count of entries
    print(f"Total entries: {len(df_reordered)}")
    
    # Save the reordered dataframe to a new CSV file
    print(f"Saving reordered labels to {output_file}...")
    df_reordered.to_csv(output_file, index=False)
    print("Done!")

if __name__ == "__main__":
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define input and output file paths
    input_file = os.path.join(current_dir, "labels.csv")
    output_file = os.path.join(current_dir, "labels_reordered.csv")
    
    reorder_labels(input_file, output_file)