# square_app.py
import streamlit as st

def main():
    st.title("Square Calculator")

    # User input
    number = st.number_input("Enter a number:", value=1.0)

    # Calculate square
    squared_value = number ** 2

    # Display result
    st.write(f"The square of {number} is {squared_value}")

if __name__ == "__main__":
    main()
