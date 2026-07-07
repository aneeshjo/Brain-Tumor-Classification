import streamlit as st


def load_css():

    st.markdown(
        """
<style>

.block-container{

    padding-top:2rem;
    padding-bottom:2rem;
}

div.stButton > button{

    width:100%;
    height:3rem;
    border-radius:12px;
    font-weight:bold;
}

div[data-testid="metric-container"]{

    border:1px solid #E5E7EB;
    padding:15px;
    border-radius:12px;
}

</style>
""",
        unsafe_allow_html=True
    )