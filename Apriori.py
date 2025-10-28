# ---------------------------------------------------------
# Association Rule Mining using Apriori Algorithm
# Supermarket Example using pandas + mlxtend
# ---------------------------------------------------------

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# --- Step 1: Create dataset as list of transactions ---
# Each transaction represents items bought together by a customer.
transactions = [
    ['Milk', 'Bread', 'Eggs'],
    ['Milk', 'Bread'],
    ['Milk', 'Cookies'],
    ['Bread', 'Butter'],
    ['Milk', 'Bread', 'Butter'],
    ['Bread', 'Eggs'],
    ['Milk', 'Butter'],
    ['Bread', 'Cookies'],
    ['Milk', 'Bread', 'Eggs', 'Butter'],
    ['Bread', 'Eggs', 'Cookies']
]

# --- Step 2: Convert the dataset into a pandas DataFrame ---
# Each column = item, each row = transaction (True if item is present)
all_items = sorted({item for trans in transactions for item in trans})

encoded_rows = []
for trans in transactions:
    encoded_rows.append({item: (item in trans) for item in all_items})

df = pd.DataFrame(encoded_rows)
print("Transaction Data:\n", df)

# --- Step 3: Apply the Apriori algorithm to find frequent itemsets ---
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)
print("\nFrequent Itemsets:\n", frequent_itemsets)

# --- Step 4: Generate association rules from frequent itemsets ---
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
print("\nAssociation Rules:\n", rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
