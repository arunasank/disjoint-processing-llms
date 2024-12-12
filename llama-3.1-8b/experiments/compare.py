import os
import pandas as pd

# Define the directories
dirs = ["test-10", "test-10-before-optimizing", "10-seed-10"]
ends_with = ["2-accuracy.csv", "-accuracy.csv", "-accuracy.csv"]

# Get the list of files matching *-accuracy.csv from each directory
files = {
    dir_: sorted([f for f in os.listdir(dir_) if f.endswith(ends_with[idx])])
    for idx, dir_ in enumerate(dirs)
}

# Initialize a list to hold the comparison results
comparison_results = []

# Iterate through files by index position
for i in range(len(files[dirs[0]])):
    file_test = os.path.join(dirs[0], files[dirs[0]][i])
    file_test_before = os.path.join(dirs[1], files[dirs[1]][i])
    file_seed = os.path.join(dirs[2], files[dirs[2]][i])

    # Read the CSV contents
    df_test = pd.read_csv(file_test)
    df_test_before = pd.read_csv(file_test_before)
    df_seed = pd.read_csv(file_seed)

    # Extract the lang and acc values
    lang = df_test.loc[0, "lang"]
    acc_test = df_test.loc[0, "acc"]
    acc_test_before = df_test_before.loc[0, "acc"]
    acc_seed = df_seed.loc[0, "acc"]

    # Append the result as a row
    comparison_results.append([lang, acc_test, acc_test_before, acc_seed])

# Create a DataFrame for the final comparison
comparison_df = pd.DataFrame(comparison_results, columns=["col", "test-10", "test-10-before-optimising", "10-seed-10"])

# Print the DataFrame as a CSV format
print(comparison_df.to_csv('2.csv', index=False))
