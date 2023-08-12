import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset from CSV
@st.cache_data
def load_data():
    return pd.read_csv("470c3ecc632925c0 (1).csv")

df = load_data()

# Drop items with recovery time of 5 days
df = df.loc[df['RecoveryTime'] != 5]

# Convert categorical columns to numerical using LabelEncoder
label_encoders = {}
for column in ['ItemType', 'LocationLost', 'TimeOfDayLost', 'ItemValue']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Safe transform function to handle unseen labels
def safe_transform(label_encoder, labels, column):
    transformed_labels = []
    for label in labels:
        if label in label_encoder.classes_:
            transformed_labels.append(label_encoder.transform([label])[0])
        else:
            # Assign the mode of the training data for unseen labels
            transformed_labels.append(df[column].mode()[0])
    return transformed_labels

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(df.drop(columns=['ItemID', 'RecoveryTime']))
y = df['RecoveryTime']

# Train a random forest regressor
model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

# Streamlit app
st.title("Lost Item Recovery Time Predictor")

# User input
st.sidebar.header("Input Lost Item Details")
item_type = st.sidebar.selectbox("ItemType", df['ItemType'].unique())
location_lost = st.sidebar.selectbox("LocationLost", df['LocationLost'].unique())
time_of_day_lost = st.sidebar.selectbox("TimeOfDayLost", df['TimeOfDayLost'].unique())
item_value = st.sidebar.selectbox("ItemValue", df['ItemValue'].unique())
previous_recoveries = st.sidebar.slider("PreviousRecoveries", 0, 10, 1)
days_since_lost = st.sidebar.slider("DaysSinceLost", 0, 30, 1)

# Predict recovery time
if st.sidebar.button("Predict Recovery Time"):
    input_data = {
        'ItemType': safe_transform(label_encoders['ItemType'], [item_type], 'ItemType'),
        'LocationLost': safe_transform(label_encoders['LocationLost'], [location_lost], 'LocationLost'),
        'TimeOfDayLost': safe_transform(label_encoders['TimeOfDayLost'], [time_of_day_lost], 'TimeOfDayLost'),
        'ItemValue': safe_transform(label_encoders['ItemValue'], [item_value], 'ItemValue'),
        'PreviousRecoveries': [previous_recoveries],
        'DaysSinceLost': [days_since_lost]
    }
    input_df = pd.DataFrame(input_data)
    input_df_scaled = scaler.transform(input_df)
    predicted_recovery_time = model.predict(input_df_scaled)
    st.write(f"Predicted Recovery Time: {predicted_recovery_time[0]:.2f} days")

st.write("## Sample Data")
st.write(df)
