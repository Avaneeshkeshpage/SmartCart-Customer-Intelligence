"""Fail-closed login for the Streamlit prototype.

Configure [auth] username/password in .streamlit/secrets.toml or use the same
keys in Streamlit Cloud Secrets. Do not commit secrets.toml to version control.
"""
import hmac
import streamlit as st


def require_login() -> bool:
    if st.session_state.get("authenticated", False):
        return True

    st.title("🛒 SmartCart Customer Intelligence")
    st.subheader("Sign in")
    try:
        expected_user = str(st.secrets["auth"]["username"])
        expected_password = str(st.secrets["auth"]["password"])
    except Exception:
        st.error("Login is not configured. Create .streamlit/secrets.toml from secrets.example.toml, set a strong password, then restart the app.")
        st.stop()

    with st.form("smartcart_login_form", clear_on_submit=False):
        username = st.text_input("Username", autocomplete="username")
        password = st.text_input("Password", type="password", autocomplete="current-password")
        submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)

    if submitted:
        user_ok = hmac.compare_digest(username.encode("utf-8"), expected_user.encode("utf-8"))
        pass_ok = hmac.compare_digest(password.encode("utf-8"), expected_password.encode("utf-8"))
        if user_ok and pass_ok:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Invalid username or password.")
    st.caption("Access is denied until valid credentials are configured and supplied.")
    return False
