LEVELS = ("transaction", "account", "customer")

TIME_COLUMN = "Transaction_Datetime"
AMOUNT_COLUMN = "Transaction_Amount"
LABEL_COLUMN = "Label"
FRAUD_TYPE_COLUMN = "Fraud_Type"

CUSTOMER_FEATURES = [
    "Customer_ID",
    "Birth_Year",
    "Gender",
    "Credit_Rating",
    "Roaming_Indicator",
    "Info_Change_Flag",
    "Loan_Type",
    "Certificate_Issued_3M_Flag",
    "Occupation_Type",
    "Income_Level",
    "Residence_Region",
    "Customer_Tenure_Months",
    "Num_Accounts",
    "Num_Cards",
    "Mobile_Carrier",
    "OTP_Registered_Flag",
    "Security_Card_Type",
    "Phone_Change_Flag",
    "Past_Fraud_Report_Count",
    "Overdue_Flag",
    "VIP_Flag",
    "Online_Banking_Joined_Flag",
    "Marketing_Consent_Flag",
]

ACCOUNT_FEATURES = [
    "Account_ID",
    "Account_Type",
    "Account_Balance",
    "Account_Creation_Date",
    "Initial_Balance",
    "Account_Suspension_Flag",
    "Daily_Transaction_Limit",
    "One_Month_Max_Amount",
    "Limit_Increased_Flag",
    "Reinstated_Flag",
    "Dormant_Days",
    "Avg_Monthly_Deposit",
    "Avg_Monthly_Withdrawal",
    "Num_Tx_Last_30d",
    "Num_Recipients_Last_30d",
    "Overdraft_Flag",
    "Linked_Card_Flag",
    "Auto_Transfer_Count",
    "Foreign_Currency_Flag",
    "Interest_Rate",
    "Account_Status",
    "Branch_Region",
    "Password_Error_Count",
    "Min_Balance_Last_30d",
]

TRANSACTION_FEATURES = [
    "Transaction_ID",
    "Transaction_Datetime",
    "Transaction_Amount",
    "Transaction_Type",
    "Transfer_Medium",
    "Time_Diff_Prev_Transaction",
    "Amount_Change_Ratio",
    "Device_Type",
    "OS_Type",
    "New_Device_Flag",
    "Device_Compromised_Flag",
    "New_Recipient_Flag",
    "Recipient_Bank_Type",
    "Logical_Distance_Prev",
    "IP_Foreign_Flag",
    "VPN_Flag",
    "Tx_Count_Last_1h",
    "Amount_Sum_Last_1h",
    "Monthly_Limit_Usage_Ratio",
    "Daily_Limit_Usage_Ratio",
    "Auth_Method",
    "Failed_Auth_Count",
    "Session_Duration_Sec",
    "Balance_After_Ratio",
    "Memo_Empty_Flag",
    "Remittance_Purpose",
    "Fee_Amount",
    "Channel_Change_Flag",
]

LEVEL_FEATURES = {
    "transaction": TRANSACTION_FEATURES,
    "account": ACCOUNT_FEATURES,
    "customer": CUSTOMER_FEATURES,
}

ID_COLUMNS = ["Customer_ID", "Account_ID", "Transaction_ID"]
DATETIME_COLUMNS = ["Account_Creation_Date", "Transaction_Datetime"]

REGIONS = ["Seoul", "Gyeonggi", "Incheon", "Busan", "Daegu", "Gwangju", "Daejeon", "Other"]

CATEGORIES = {
    "Gender": ["M", "F"],
    "Loan_Type": ["No_Loan", "Mortgage", "Credit", "Auto", "Student"],
    "Occupation_Type": ["Office", "Self_Employed", "Student", "Retired", "Public", "Other"],
    "Residence_Region": REGIONS,
    "Mobile_Carrier": ["SKT", "KT", "LGU", "MVNO"],
    "Security_Card_Type": ["OTP", "Security_Card", "Mobile_Cert"],
    "Account_Type": ["Checking", "Savings", "Deposit", "Loan"],
    "Account_Status": ["Active", "Dormant", "Restricted"],
    "Branch_Region": REGIONS,
    "Transaction_Type": ["Transfer", "Payment", "Withdrawal", "Deposit"],
    "Transfer_Medium": ["Mobile", "Internet", "ATM", "Branch", "Telephone"],
    "Device_Type": ["Mobile", "PC", "ATM_Terminal", "Teller"],
    "OS_Type": ["Android", "iOS", "Windows", "MacOS", "Other"],
    "Recipient_Bank_Type": ["Same", "Other", "No_Recipient"],
    "Auth_Method": ["OTP", "Biometric", "Certificate", "Password"],
    "Remittance_Purpose": ["Living", "Business", "Loan", "Investment", "Other"],
}

MONETARY_COLUMNS = [
    "Account_Balance",
    "Initial_Balance",
    "Daily_Transaction_Limit",
    "One_Month_Max_Amount",
    "Avg_Monthly_Deposit",
    "Avg_Monthly_Withdrawal",
    "Min_Balance_Last_30d",
    "Transaction_Amount",
    "Amount_Sum_Last_1h",
]

FRAUD_TYPES = {
    "a": ("Large logical distance between consecutive transactions", 0.10),
    "b": ("Transactions with compromised devices", 0.18),
    "c": ("Transactions from a previously unused device", 0.09),
    "d": ("ATM withdrawals after a long period of inactivity", 0.12),
    "e": ("Transactions from recently reinstated accounts", 0.20),
    "f": ("Transactions with unusually high amounts", 0.16),
    "g": ("Multiple transfers within a short time window", 0.08),
    "h": ("Transactions during early morning hours (1-6 AM)", 0.14),
    "i": ("Transfers to previously unused recipient accounts", 0.07),
    "j": ("Transactions from older customers (60+ years)", 0.11),
    "k": ("Abnormal transactions exceeding 1-month limits", 0.18),
    "l": ("Small-scale fraud attempts (phishing, account takeover)", 0.12),
}
NO_FRAUD_TYPE = "-"


def numeric_columns(level):
    excluded = set(ID_COLUMNS) | set(DATETIME_COLUMNS) | set(CATEGORIES)
    return [column for column in LEVEL_FEATURES[level] if column not in excluded]


def categorical_columns(level):
    return [column for column in LEVEL_FEATURES[level] if column in CATEGORIES]


def validate_transactions(df, require_labels=False):
    """Validate the public input contract before any label-dependent processing."""
    import numpy as np
    import pandas as pd
    required = set(CUSTOMER_FEATURES + ACCOUNT_FEATURES + TRANSACTION_FEATURES) - set(ID_COLUMNS)
    if require_labels:
        required.add(LABEL_COLUMN)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"missing transaction columns: {missing}")
    if require_labels and len(df) == 0:
        raise ValueError("training transactions cannot be empty")
    for column in DATETIME_COLUMNS:
        if not pd.api.types.is_datetime64_any_dtype(df[column]) or df[column].isna().any():
            raise ValueError(f"{column} must contain valid pandas datetimes")
    amounts = df[AMOUNT_COLUMN].to_numpy(dtype=float)
    if not np.isfinite(amounts).all() or (amounts < 0).any():
        raise ValueError("transaction amounts must be finite and nonnegative")
    if require_labels and not np.isin(df[LABEL_COLUMN], [0, 1]).all():
        raise ValueError("Label must contain only 0 and 1")
