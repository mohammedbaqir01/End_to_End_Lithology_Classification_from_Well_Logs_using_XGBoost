# Cleans training and test well-log data by removing unused curves, requiring core logs, and interpolating density correction values per well.


def preprocess_data(train_df, test_df):

    # will drop the columns that are not required for the model training
    cols_to_drop = [
        "SGR",
        "RMIC",
        "MUDWEIGHT",
        "DCAL",
        "RXO",
        "DTS",
        "ROPA",
        "ROP",
        "SP",
        "RSHA",
    ]

    train_df = train_df.drop(columns=cols_to_drop, errors="ignore")
    test_df = test_df.drop(columns=cols_to_drop, errors="ignore")

    # Handling training data first, we will drop the rows with missing values in the required cloumns
    core_required = ["CALI", "RDEP", "DTC", "RHOB", "NPHI"]
    train_df.dropna(subset=["GR"] + core_required, inplace=True)

    train_df["DRHO"] = train_df.groupby("well_name")["DRHO"].transform(
        lambda x: x.interpolate()
    )

    # Handling testing data, we will drop the rows with missing values in the required cloumns
    core_required = ["CALI", "RDEP", "DTC", "RHOB", "NPHI"]
    test_df.dropna(subset=["GR"] + core_required, inplace=True)

    test_df["DRHO"] = test_df.groupby("well_name")["DRHO"].transform(
        lambda x: x.interpolate()
    )

    return train_df, test_df
