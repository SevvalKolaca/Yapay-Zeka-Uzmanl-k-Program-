from app import App
import streamlit as st

def main():

    st.set_page_config(page_title="Breastcare ML")

    app = App()
    app.run()

if __name__ == "__main__":
    main()
