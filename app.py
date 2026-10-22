import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="How Danceable Is This Song?", layout="wide")
st.title("How Danceable Is This Song?")
st.caption("Upload a CSV file, look at the data, draw a chart, and make a prediction.")

# The one sentence the assignment asks for.
st.write("This model helps build party playlists by predicting danceability, and if it's wrong, a song could clear the dance floor instead of filling it.")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 3:
        st.error("This file needs at least three numeric columns.")
        st.stop()

    # Part 1: preview the data.
    st.subheader("1. Preview")
    st.write(f"{len(df):,} rows and {len(df.columns)} columns.")
    st.dataframe(df.head(10), hide_index=True, width="stretch")

    # Part 2: pick a column and a chart type, then draw it.
    st.subheader("2. Chart")
    c1, c2 = st.columns(2)
    chart_type = c1.selectbox("Chart type", ["Histogram", "Scatter", "Line"])
    column = c2.selectbox("Column", numeric_cols)

    if chart_type == "Histogram":
        fig = px.histogram(df, x=column, title=f"Distribution of {column}")
    else:
        # Scatter and line need a second column for the x axis.
        x_options = [c for c in df.columns if c != column]
        x_axis = st.selectbox("X axis", x_options)
        if chart_type == "Scatter":
            fig = px.scatter(df, x=x_axis, y=column, title=f"{column} versus {x_axis}")
        else:
            # Sort by the x axis so the line does not zigzag.
            fig = px.line(df.sort_values(x_axis), x=x_axis, y=column, title=f"{column} by {x_axis}")
    st.plotly_chart(fig, width="stretch")

    # Part 3: pick two inputs and a target, fit a linear regression.
    st.subheader("3. Prediction")
    st.write("For the Spotify file, try danceability as the target, with valence and acousticness as the inputs.")
    c1, c2, c3 = st.columns(3)
    target = c1.selectbox("Target: what you want to predict", numeric_cols)
    input_options = [c for c in numeric_cols if c != target]
    input1 = c2.selectbox("Input 1", input_options)
    input2_options = [c for c in input_options if c != input1]
    input2 = c3.selectbox("Input 2", input2_options)

    inputs = [input1, input2]
    data = df[inputs + [target]].dropna()
    if len(data) < 5:
        st.error("Need at least 5 rows with values in all three columns.")
        st.stop()

    x = data[inputs]
    y = data[target]
    model = LinearRegression()
    model.fit(x, y)

    m1, m2, m3 = st.columns(3)
    m1.metric(f"Each extra {input1} adds", f"{model.coef_[0]:,.3f}")
    m2.metric(f"Each extra {input2} adds", f"{model.coef_[1]:,.3f}")
    m3.metric("R2 on this data", f"{model.score(x, y):.2f}")

    # The user types a value for each input.
    st.write("Type in values and see what the model predicts.")
    v1, v2 = st.columns(2)
    value1 = v1.number_input(f"Value for {input1}", value=float(data[input1].mean()))
    value2 = v2.number_input(f"Value for {input2}", value=float(data[input2].mean()))

    # Same column names, same order as training.
    new_row = pd.DataFrame({input1: [value1], input2: [value2]})[inputs]
    prediction = model.predict(new_row)[0]
    st.metric(f"Predicted {target}", f"{prediction:,.2f}")

    # Warn the user if a value is outside the range the model learned from.
    low1, high1 = data[input1].min(), data[input1].max()
    if value1 < low1 or value1 > high1:
        st.warning(f"{input1} is outside the range in the data ({low1:,.2f} to {high1:,.2f}). Treat the prediction with caution.")

    low2, high2 = data[input2].min(), data[input2].max()
    if value2 < low2 or value2 > high2:
        st.warning(f"{input2} is outside the range in the data ({low2:,.2f} to {high2:,.2f}). Treat the prediction with caution.")
else:
    st.info("Upload a CSV file to get started. It needs at least three numeric columns.")
