
import os
import requests
import streamlit as st

st.set_page_config(
    page_title="CreditIQ",
    page_icon="📊",
    layout="wide",
)

st.title("📊 CreditIQ: Credit Default Scoring")
st.caption("Educational model, not a lending tool.")

# Connect to deployed FastAPI backend
api = st.sidebar.text_input(
    "API URL",
    os.environ.get(
        "CREDITIQ_API_URL",
        "https://creditiq-vbx5.onrender.com",
    ),
).strip().rstrip("/")

with st.form("applicant"):
    c1, c2 = st.columns(2)

    loan_amnt = c1.number_input(
        "Loan amount", 500, 100000, 12000, step=500
    )
    term_months = c2.selectbox("Term (months)", [36, 60])

    installment = c1.number_input(
        "Monthly installment", 10.0, 10000.0, 400.0
    )
    annual_inc = c2.number_input(
        "Annual income", 0, 10_000_000, 60000, step=1000
    )

    dti = c1.number_input(
        "Debt-to-income (%)", 0.0, 100.0, 18.0
    )
    revol_util = c2.number_input(
        "Credit card utilization (%)", 0.0, 100.0, 45.0
    )

    revol_bal = c1.number_input(
        "Revolving balance", 0, 1_000_000, 8000, step=500
    )
    fico = c2.number_input("Credit score (FICO)", 300, 846, 690)

    delinq = c1.number_input(
        "Missed payments, last 2 years", 0, 50, 0
    )
    pub_rec = c2.number_input("Public records", 0, 20, 0)

    open_acc = c1.number_input("Open credit lines", 0, 80, 9)
    total_acc = c2.number_input("Total credit accounts", 0, 150, 20)

    emp_years = c1.number_input("Years employed", 0, 10, 3)
    history = c2.number_input(
        "Credit history (years)", 0.0, 80.0, 12.0
    )

    home = c1.selectbox(
        "Home ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"]
    )
    verif = c2.selectbox(
        "Income verification",
        ["Verified", "Source Verified", "Not Verified"],
    )

    purpose = st.selectbox(
        "Loan purpose",
        [
            "debt_consolidation",
            "credit_card",
            "home_improvement",
            "car",
            "small_business",
            "other",
        ],
    )

    go = st.form_submit_button(
        "Score applicant",
        use_container_width=True,
    )

if go:
    payload = {
        "loan_amnt": loan_amnt,
        "term_months": term_months,
        "installment": installment,
        "annual_inc": annual_inc,
        "dti": dti,
        "revol_util": revol_util,
        "revol_bal": revol_bal,
        "delinq_2yrs": delinq,
        "open_acc": open_acc,
        "pub_rec": pub_rec,
        "total_acc": total_acc,
        "fico_range_low": fico,
        "fico_range_high": min(fico + 4, 850),
        "emp_years": emp_years,
        "credit_history_years": history,
        "home_ownership": home,
        "verification_status": verif,
        "purpose": purpose,
    }

    try:
        with st.spinner("Getting prediction..."):
            response = requests.post(
                f"{api}/score",
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            res = response.json()

        if res.get("warning"):
            st.warning(res["warning"])

        decision = res.get("decision", "unknown").lower()

        if decision == "approve":
            st.success("Model decision: APPROVE")
        elif decision == "review":
            st.warning("Model decision: REVIEW")
        elif decision == "decline":
            st.error("Model decision: DECLINE")
        else:
            st.info(f"Model decision: {decision.upper()}")

        c1, c2 = st.columns(2)
        c1.metric(
            "Default probability",
            f"{float(res['default_probability']):.1%}",
        )
        c2.metric("Risk band", res["risk_band"])

        st.subheader("Main reasons for this outcome")
        for reason in res.get("top_risk_factors", []):
            st.write(f"- {reason}")

        st.caption(
            f"Decline cutoff: {float(res['cutoff']):.1%} "
            f"| Model: {res['model']}"
        )
        st.info(res["disclaimer"])

    except requests.exceptions.Timeout:
        st.error("API response timed out. Please try again.")
    except requests.exceptions.HTTPError as e:
        st.error(f"API returned an HTTP error: {e}")
        if e.response is not None:
            st.code(e.response.text)
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to API: {type(e).__name__}")
    except (KeyError, ValueError, TypeError) as e:
        st.error(f"Unexpected API response: {e}")