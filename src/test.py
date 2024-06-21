from data import sample_data_remotely

if __name__ == "__main__":
    df = sample_data_remotely()
    print(df.head())