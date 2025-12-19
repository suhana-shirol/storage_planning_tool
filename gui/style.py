import streamlit as st


def apply_brand_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bmw-text: #031E49;
            --bmw-blue: #DAECFF;
            --bmw-info: #DAECFF;
            --bmw-field-bg: #f2f4f5;
        }

        body, .stApp, .stMarkdown, .stText, div, label, span, p,
        h1, h2, h3, h4, h5, h6 {
            color: #031E49 !important;
            font-family: "Helvetica", "Arial", sans-serif;
        }

        .stApp {
            background-color: #FFFFFF !important;
        }

        [data-testid="stSidebar"], [data-testid="collapsedControl"] {
            display: none !important;
        }

        a, a:visited {
            color: #DAECFF !important;
        }
        a:hover {
            color: #014a80 !important;
        }

        [data-testid="stNumberInput"] input,
        [data-testid="stNumberInput"] input:focus,
        [data-testid="stTextInput"] input,
        [data-testid="stTextInput"] input:focus,
        [data-testid="stSelectbox"] div[role="combobox"],
        [data-testid="stSelectbox"] div[role="combobox"]:focus,
        [data-testid="stSelectbox"] > div > div,
        [data-testid="stMultiselect"] div[role="combobox"],
        [data-testid="stMultiselect"] div[role="combobox"]:focus,
        div[data-baseweb="textarea"] textarea,
        div[data-baseweb="textarea"] textarea:focus {
            background-color: var(--bmw-field-bg) !important;
            color: #031E49 !important;
        }

        .bmw-field-wrapper {
            background-color: var(--bmw-field-bg) !important;
            border-radius: 6px;
            padding: 0.5rem;
        }

        .bmw-header {
            position: sticky;
            top: 0;
            z-index: 100;
            background-color: var(--bmw-field-bg);
            border-bottom: 1px solid #d8d8d8;
            padding: 0.5rem 1rem;
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        .bmw-header img {
            height: 32px;
        }
        .bmw-header h1 {
            font-size: 1.1rem;
            margin: 0;
            color: var(--bmw-text);
        }

        [data-testid="stNumberInput"] button {
            background-color: #f2f4f5 !important;
            color: #031E49 !important;
            border: none !important;
        }
        [data-testid="stNumberInput"] button:hover {
            background-color: #031E49 !important;
            color: #FFFFFF !important;
        }

        .stButton button, .stDownloadButton button, .stFormSubmitButton button {
            background-color: #031E49 !important;
            border: none !important;
            color: #FFFFFF !important;
        }
        .stButton button *, .stDownloadButton button *, .stFormSubmitButton button * {
            color: #FFFFFF !important;
        }
        .stButton button:hover, .stDownloadButton button:hover, .stFormSubmitButton button:hover {
            background-color: #193a72 !important;
            color: #FFFFFF !important;
        }
        .stButton button:hover *, .stDownloadButton button:hover *, .stFormSubmitButton button:hover * {
            color: #FFFFFF !important;
        }

        div[data-testid="stMultiSelect"] div[data-baseweb="tag"],
        div[data-testid="stMultiSelect"] div[data-baseweb="tag"] > div,
        div[data-testid="stMultiSelect"] div[data-baseweb="tag"] span {
            background-color: var(--bmw-info) !important;
            color: #FFFFFF !important;
            border: none !important;
        }

        .stAlert[data-testid="stAlert-info"] {
            background-color: var(--bmw-info) !important;
            color: var(--bmw-blue) !important;
            border: 1px solid var(--bmw-blue) !important;
        }

        .stAlert[data-testid="stAlert-warning"] {
            background-color: #feecec !important;
            color: #EE0405 !important;
            border: 1px solid #EE0405 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
