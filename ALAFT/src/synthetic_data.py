from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schema import (
    ACCOUNT_FEATURES,
    CATEGORIES,
    CUSTOMER_FEATURES,
    FRAUD_TYPE_COLUMN,
    FRAUD_TYPES,
    LABEL_COLUMN,
    NO_FRAUD_TYPE,
    REGIONS,
    TIME_COLUMN,
    TRANSACTION_FEATURES,
)

SLOT_MINUTES = 30
SLOTS_PER_DAY = 24 * 60 // SLOT_MINUTES
EARLY_MORNING_HOURS = range(1, 6)


@dataclass(frozen=True)
class SyntheticBankConfig:
    start: str = "2021-11-30"
    n_days: int = 40
    n_customers: int = 1050
    n_accounts: int = 8400
    transactions_per_slot: float = 30.0
    background_fraud_rate: float = 0.002
    campaign_fraud_rate: float = 0.2
    campaign_interval_hours: float = 30.0
    victim_customer_rate: float = 0.1
    unlabeled_fraud_rate: float = 0.2
    seed: int = 42


def generate_synthetic_bank(config=SyntheticBankConfig()):
    rng = np.random.default_rng(config.seed)
    customers, victims = _customers(rng, config)
    accounts = _accounts(rng, config, customers, victims)
    customers["Num_Accounts"] = accounts["Customer_ID"].value_counts().reindex(customers["Customer_ID"]).to_numpy()

    timestamps, fraud_type = _transaction_times(rng, config)
    account_index = _transaction_accounts(rng, fraud_type, accounts, customers, victims)
    transactions = _transactions(rng, timestamps, fraud_type, accounts.iloc[account_index])

    is_fraud = fraud_type != NO_FRAUD_TYPE
    unlabeled = is_fraud & (rng.random(len(fraud_type)) < config.unlabeled_fraud_rate)
    transactions[LABEL_COLUMN] = (is_fraud & ~unlabeled).astype(int)
    transactions[FRAUD_TYPE_COLUMN] = np.where(unlabeled, NO_FRAUD_TYPE, fraud_type)

    data = transactions.merge(accounts, on="Account_ID").merge(customers, on="Customer_ID")
    data = data.sort_values(TIME_COLUMN, kind="stable").reset_index(drop=True)
    data["Transaction_ID"] = [f"TX{i:08d}" for i in range(len(data))]
    return data[CUSTOMER_FEATURES + ACCOUNT_FEATURES + TRANSACTION_FEATURES + [LABEL_COLUMN, FRAUD_TYPE_COLUMN]]


def _flag(rng, probability):
    return (rng.random(len(probability)) < probability).astype(int)


def _customers(rng, config):
    n = config.n_customers
    victim = rng.random(n) < config.victim_customer_rate
    risk = lambda base, elevated: np.where(victim, elevated, base)

    customers = pd.DataFrame({
        "Customer_ID": [f"CUST{i:05d}" for i in range(n)],
        "Birth_Year": rng.integers(1940, 2005, n),
        "Gender": rng.choice(CATEGORIES["Gender"], n),
        "Credit_Rating": np.clip(rng.normal(720, 90, n) - 60 * victim, 300, 1000).round().astype(int),
        "Roaming_Indicator": _flag(rng, risk(0.05, 0.25)),
        "Info_Change_Flag": _flag(rng, risk(0.04, 0.35)),
        "Loan_Type": rng.choice(CATEGORIES["Loan_Type"], n, p=[0.55, 0.2, 0.15, 0.06, 0.04]),
        "Certificate_Issued_3M_Flag": _flag(rng, risk(0.10, 0.50)),
        "Occupation_Type": rng.choice(CATEGORIES["Occupation_Type"], n, p=[0.4, 0.15, 0.1, 0.15, 0.1, 0.1]),
        "Income_Level": rng.integers(1, 11, n),
        "Residence_Region": rng.choice(REGIONS, n),
        "Customer_Tenure_Months": np.where(victim, rng.integers(1, 120, n), rng.integers(1, 361, n)),
        "Num_Cards": rng.integers(0, 7, n),
        "Mobile_Carrier": np.where(
            victim,
            rng.choice(CATEGORIES["Mobile_Carrier"], n, p=[0.3, 0.2, 0.2, 0.3]),
            rng.choice(CATEGORIES["Mobile_Carrier"], n, p=[0.45, 0.3, 0.2, 0.05]),
        ),
        "OTP_Registered_Flag": _flag(rng, risk(0.6, 0.35)),
        "Security_Card_Type": rng.choice(CATEGORIES["Security_Card_Type"], n),
        "Phone_Change_Flag": _flag(rng, risk(0.03, 0.30)),
        "Past_Fraud_Report_Count": rng.poisson(risk(0.05, 0.6)),
        "Overdue_Flag": _flag(rng, risk(0.05, 0.12)),
        "VIP_Flag": _flag(rng, np.full(n, 0.05)),
        "Online_Banking_Joined_Flag": _flag(rng, np.full(n, 0.85)),
        "Marketing_Consent_Flag": _flag(rng, np.full(n, 0.5)),
    })
    return customers, victim


def _accounts(rng, config, customers, victims):
    n = config.n_accounts
    owner = np.concatenate([np.arange(config.n_customers), rng.integers(0, config.n_customers, n - config.n_customers)])
    victim = victims[owner]
    risk = lambda base, elevated: np.where(victim, elevated, base)

    balance = rng.lognormal(np.log(1.5e7), 1.2, n)
    dormant = rng.random(n) < 0.05
    dormant_days = np.where(dormant, rng.uniform(180, 900, n), rng.exponential(15, n)).round()
    suspension = _flag(rng, risk(0.01, 0.10))
    status = np.select([suspension == 1, dormant_days > 180], ["Restricted", "Dormant"], "Active")
    start = pd.Timestamp(config.start)

    return pd.DataFrame({
        "Account_ID": [f"ACC{i:06d}" for i in range(n)],
        "Customer_ID": customers["Customer_ID"].to_numpy()[owner],
        "Account_Type": rng.choice(CATEGORIES["Account_Type"], n, p=[0.5, 0.3, 0.15, 0.05]),
        "Account_Balance": balance.round(-1),
        "Account_Creation_Date": start - pd.to_timedelta(rng.integers(30, 8000, n), unit="D"),
        "Initial_Balance": rng.lognormal(np.log(1e6), 1.0, n).round(-1),
        "Account_Suspension_Flag": suspension,
        "Daily_Transaction_Limit": rng.choice([1e6, 5e6, 1e7, 5e7], n, p=[0.2, 0.4, 0.3, 0.1]),
        "One_Month_Max_Amount": rng.lognormal(np.log(5e6), 0.8, n).round(-1),
        "Limit_Increased_Flag": _flag(rng, risk(0.05, 0.30)),
        "Reinstated_Flag": _flag(rng, risk(0.02, 0.20)),
        "Dormant_Days": dormant_days,
        "Avg_Monthly_Deposit": rng.lognormal(np.log(3e6), 0.9, n).round(-1),
        "Avg_Monthly_Withdrawal": rng.lognormal(np.log(2.8e6), 0.9, n).round(-1),
        "Num_Tx_Last_30d": rng.poisson(25, n),
        "Num_Recipients_Last_30d": rng.poisson(risk(6, 10)),
        "Overdraft_Flag": _flag(rng, np.full(n, 0.03)),
        "Linked_Card_Flag": _flag(rng, np.full(n, 0.7)),
        "Auto_Transfer_Count": rng.poisson(2, n),
        "Foreign_Currency_Flag": _flag(rng, np.full(n, 0.05)),
        "Interest_Rate": rng.uniform(0.1, 4.0, n).round(2),
        "Account_Status": status,
        "Branch_Region": rng.choice(REGIONS, n),
        "Password_Error_Count": rng.poisson(risk(0.1, 1.0)),
        "Min_Balance_Last_30d": (balance * rng.uniform(0.1, 1.0, n)).round(-1),
    })


def _campaign_slots(rng, n_slots, interval_slots, type_names, type_rates):
    active = np.zeros(n_slots, dtype=bool)
    dominant = np.full(n_slots, NO_FRAUD_TYPE, dtype=object)
    t = rng.gamma(4.0, interval_slots / 4.0)
    while t < n_slots:
        start, duration = int(t), int(rng.integers(1, 4))
        active[start:start + duration] = True
        dominant[start:start + duration] = rng.choice(type_names, p=type_rates)
        t += duration + rng.gamma(4.0, interval_slots / 4.0)
    return active, dominant


def _transaction_times(rng, config):
    n_slots = config.n_days * SLOTS_PER_DAY
    slot_starts = pd.date_range(config.start, periods=n_slots, freq=f"{SLOT_MINUTES}min")
    hours = slot_starts.hour.to_numpy() + slot_starts.minute.to_numpy() / 60
    profile = 0.25 + np.exp(-(((hours - 14) / 5) ** 2))
    volume = config.transactions_per_slot * profile / profile.mean()

    type_names = np.array(list(FRAUD_TYPES))
    type_rates = np.array([rate for _, rate in FRAUD_TYPES.values()])
    type_rates = type_rates / type_rates.sum()
    campaign, dominant = _campaign_slots(rng, n_slots, config.campaign_interval_hours * 2, type_names, type_rates)

    n_normal = rng.poisson(volume)
    n_background = rng.poisson(volume * config.background_fraud_rate)
    campaign_share = config.campaign_fraud_rate / (1 - config.campaign_fraud_rate)
    n_campaign = np.where(campaign, rng.poisson(volume * campaign_share) + 2, 0)

    slot_seconds = SLOT_MINUTES * 60
    burst_centers = rng.uniform(0, slot_seconds, (n_slots, 2))

    normal_slot = np.repeat(np.arange(n_slots), n_normal)
    background_slot = np.repeat(np.arange(n_slots), n_background)
    campaign_slot = np.repeat(np.arange(n_slots), n_campaign)

    campaign_offset = burst_centers[campaign_slot, rng.integers(0, 2, len(campaign_slot))]
    campaign_offset = np.clip(campaign_offset + rng.normal(0, 240, len(campaign_slot)), 0, slot_seconds - 1)
    campaign_type = np.where(
        rng.random(len(campaign_slot)) < 0.6,
        dominant[campaign_slot],
        rng.choice(type_names, len(campaign_slot), p=type_rates),
    )

    slot = np.concatenate([normal_slot, background_slot, campaign_slot])
    offset = np.concatenate([
        rng.uniform(0, slot_seconds, len(normal_slot) + len(background_slot)),
        campaign_offset,
    ])
    fraud_type = np.concatenate([
        np.full(len(normal_slot), NO_FRAUD_TYPE, dtype=object),
        rng.choice(type_names, len(background_slot), p=type_rates),
        campaign_type,
    ])
    timestamps = slot_starts[slot] + pd.to_timedelta(offset.round(), unit="s")

    non_h = type_names != "h"
    misplaced_h = (fraud_type == "h") & ~np.isin(timestamps.hour, list(EARLY_MORNING_HOURS))
    fraud_type[misplaced_h] = rng.choice(type_names[non_h], misplaced_h.sum(), p=type_rates[non_h] / type_rates[non_h].sum())
    return timestamps, fraud_type


def _transaction_accounts(rng, fraud_type, accounts, customers, victims):
    n_accounts = len(accounts)
    account_index = rng.integers(0, n_accounts, len(fraud_type))

    victim_ids = set(customers.loc[victims, "Customer_ID"])
    older_ids = set(customers.loc[customers["Birth_Year"] <= 1964, "Customer_ID"])
    owner = accounts["Customer_ID"]
    pools = {
        "d": np.flatnonzero(accounts["Dormant_Days"] > 180),
        "e": np.flatnonzero(accounts["Reinstated_Flag"] == 1),
        "j": np.flatnonzero(owner.isin(older_ids)),
    }
    victim_pool = np.flatnonzero(owner.isin(victim_ids))

    for fraud in FRAUD_TYPES:
        rows = np.flatnonzero(fraud_type == fraud)
        pool = pools.get(fraud, victim_pool)
        from_pool = rng.random(len(rows)) < (1.0 if fraud in pools else 0.7)
        account_index[rows[from_pool]] = rng.choice(pool, from_pool.sum())
    return account_index


def _transactions(rng, timestamps, fraud_type, accounts):
    n = len(fraud_type)
    is_fraud = fraud_type != NO_FRAUD_TYPE

    tx = pd.DataFrame({
        "Account_ID": accounts["Account_ID"].to_numpy(),
        "Transaction_Datetime": timestamps,
        "Transaction_Amount": rng.lognormal(np.log(3e5), 1.1, n),
        "Transaction_Type": rng.choice(CATEGORIES["Transaction_Type"], n, p=[0.5, 0.25, 0.15, 0.1]),
        "Transfer_Medium": rng.choice(CATEGORIES["Transfer_Medium"], n, p=[0.55, 0.2, 0.15, 0.07, 0.03]),
        "Time_Diff_Prev_Transaction": rng.exponential(600, n),
        "Amount_Change_Ratio": rng.lognormal(0, 0.5, n),
        "New_Device_Flag": _flag(rng, np.full(n, 0.03)),
        "Device_Compromised_Flag": _flag(rng, np.full(n, 0.002)),
        "New_Recipient_Flag": _flag(rng, np.full(n, 0.15)),
        "Recipient_Bank_Type": rng.choice(CATEGORIES["Recipient_Bank_Type"], n, p=[0.45, 0.45, 0.1]),
        "Logical_Distance_Prev": rng.exponential(5, n),
        "IP_Foreign_Flag": _flag(rng, np.full(n, 0.01)),
        "VPN_Flag": _flag(rng, np.full(n, 0.01)),
        "Tx_Count_Last_1h": rng.poisson(0.5, n),
        "Monthly_Limit_Usage_Ratio": rng.beta(2, 5, n),
        "Auth_Method": rng.choice(CATEGORIES["Auth_Method"], n, p=[0.35, 0.35, 0.2, 0.1]),
        "Failed_Auth_Count": rng.poisson(0.05, n),
        "Session_Duration_Sec": rng.lognormal(np.log(120), 0.6, n),
        "Balance_After_Ratio": rng.uniform(0, 1, n),
        "Memo_Empty_Flag": _flag(rng, np.full(n, 0.3)),
        "Remittance_Purpose": rng.choice(CATEGORIES["Remittance_Purpose"], n, p=[0.5, 0.2, 0.1, 0.1, 0.1]),
        "Fee_Amount": rng.choice([0, 500, 1000], n, p=[0.6, 0.3, 0.1]),
        "Channel_Change_Flag": _flag(rng, np.full(n, 0.05)),
    })
    atm = tx["Transfer_Medium"] == "ATM"
    tx.loc[atm, "Transaction_Type"] = rng.choice(["Withdrawal", "Deposit"], atm.sum(), p=[0.8, 0.2])

    _inject_fraud_patterns(rng, tx, fraud_type, is_fraud)
    _derive_dependent_features(rng, tx, accounts)
    return tx


def _inject_fraud_patterns(rng, tx, fraud_type, is_fraud):
    def override(mask, column, values):
        tx.loc[mask, column] = values

    def raise_flag(mask, column, probability):
        tx.loc[mask, column] = np.maximum(tx.loc[mask, column], _flag(rng, np.full(mask.sum(), probability)))

    raise_flag(is_fraud, "New_Recipient_Flag", 0.5)
    raise_flag(is_fraud, "Memo_Empty_Flag", 0.7)
    override(is_fraud, "Session_Duration_Sec", tx.loc[is_fraud, "Session_Duration_Sec"] * 0.5)
    override(is_fraud, "Balance_After_Ratio", rng.uniform(0, 0.3, is_fraud.sum()))
    override(is_fraud, "Transaction_Amount", tx.loc[is_fraud, "Transaction_Amount"] * rng.lognormal(0.7, 0.5, is_fraud.sum()))

    of_type = {fraud: fraud_type == fraud for fraud in FRAUD_TYPES}
    size = {fraud: mask.sum() for fraud, mask in of_type.items()}

    override(of_type["a"], "Logical_Distance_Prev", 100 + rng.exponential(400, size["a"]))
    raise_flag(of_type["a"], "IP_Foreign_Flag", 0.4)

    raise_flag(of_type["b"], "Device_Compromised_Flag", 0.85)
    raise_flag(of_type["b"], "VPN_Flag", 0.3)

    raise_flag(of_type["c"], "New_Device_Flag", 0.9)
    raise_flag(of_type["c"], "Channel_Change_Flag", 0.5)

    override(of_type["d"], "Transfer_Medium", "ATM")
    override(of_type["d"], "Transaction_Type", "Withdrawal")
    override(of_type["d"], "Time_Diff_Prev_Transaction", rng.uniform(180, 900, size["d"]) * 24 * 60)
    override(of_type["d"], "Transaction_Amount", rng.lognormal(np.log(1.2e7), 0.4, size["d"]))

    override(of_type["e"], "Auth_Method", rng.choice(["Password", "OTP"], size["e"]))

    override(of_type["f"], "Transaction_Amount", rng.lognormal(np.log(3e7), 0.6, size["f"]))
    override(of_type["f"], "Amount_Change_Ratio", rng.lognormal(np.log(8), 0.5, size["f"]))

    override(of_type["g"], "Tx_Count_Last_1h", rng.poisson(6, size["g"]) + 3)
    override(of_type["g"], "Time_Diff_Prev_Transaction", rng.exponential(3, size["g"]))

    override(of_type["h"], "Session_Duration_Sec", rng.lognormal(np.log(30), 0.4, size["h"]))

    override(of_type["i"], "New_Recipient_Flag", 1)
    override(of_type["i"], "Recipient_Bank_Type", "Other")

    override(of_type["j"], "Transfer_Medium", rng.choice(["Telephone", "Mobile"], size["j"]))
    override(of_type["j"], "Transaction_Amount", tx.loc[of_type["j"], "Transaction_Amount"] * 2)

    override(of_type["k"], "Monthly_Limit_Usage_Ratio", rng.uniform(1.0, 2.5, size["k"]))

    override(of_type["l"], "Transaction_Amount", rng.lognormal(np.log(3e4), 0.5, size["l"]))
    override(of_type["l"], "Failed_Auth_Count", rng.poisson(2, size["l"]) + 1)


def _derive_dependent_features(rng, tx, accounts):
    medium = tx["Transfer_Medium"]
    tx["Device_Type"] = np.select(
        [medium == "ATM", medium == "Branch", medium == "Internet"],
        ["ATM_Terminal", "Teller", "PC"],
        "Mobile",
    )
    tx["OS_Type"] = np.select(
        [tx["Device_Type"] == "Mobile", tx["Device_Type"] == "PC"],
        [rng.choice(["Android", "iOS"], len(tx), p=[0.6, 0.4]), rng.choice(["Windows", "MacOS"], len(tx), p=[0.85, 0.15])],
        "Other",
    )
    amount = tx["Transaction_Amount"].clip(1_000, 5e8).round(-1)
    tx["Transaction_Amount"] = amount
    tx["Amount_Sum_Last_1h"] = (amount * (1 + tx["Tx_Count_Last_1h"]) * rng.uniform(0.5, 1.5, len(tx))).round(-1)
    tx["Daily_Limit_Usage_Ratio"] = (amount / accounts["Daily_Transaction_Limit"].to_numpy()).clip(0, 3)
    tx["Time_Diff_Prev_Transaction"] = tx["Time_Diff_Prev_Transaction"].round(1)
