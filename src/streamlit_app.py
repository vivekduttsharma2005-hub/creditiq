import streamlit as st

st.set_page_config(page_title="CreditIQ", page_icon="📊")
st.title("CreditIQ: credit default scoring")
st.caption("Educational model, not a lending tool.")
api = st.sidebar.text_input("API URL", os.environ.get("CREDITIQ_API_URL", "http://localhost:8000"))

with st.form("applicant"):
    c1, c2 = st.columns(2)
    loan_amnt = c1.number_input("Loan amount", 500, 100000, 12000, step=500)
    term_months = c2.selectbox("Term (months)", [36, 60])
    installment = c1.number_input("Monthly installment", 10.0, 10000.0, 400.0)
    annual_inc = c2.number_input("Annual income", 0, 10_000_000, 60000, step=1000)
    dti = c1.number_input("Debt-to-income (%)", 0.0, 100.0, 18.0)
    revol_util = c2.number_input("Credit card utilization (%)", 0.0, 100.0, 45.0)
    revol_bal = c1.number_input("Revolving balance", 0, 1_000_000, 8000, step=500)
    fico = c2.number_input("Credit score (FICO)", 300, 846, 690)
    delinq = c1.number_input("Missed payments, last 2 years", 0, 50, 0)
    pub_rec = c2.number_input("Public records", 0, 20, 0)
    open_acc = c1.number_input("Open credit lines", 0, 80, 9)
    total_acc = c2.number_input("Total credit accounts", 0, 150, 20)
    emp_years = c1.number_input("Years employed", 0, 10, 3)
    history = c2.number_input("Credit history (years)", 0.0, 80.0, 12.0)
    home = c1.selectbox("Home ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
    verif = c2.selectbox("Income verification", ["Verified", "Source Verified", "Not Verified"])
    purpose = st.selectbox("Loan purpose", ["debt_consolidation", "credit_card", "home_improvement",
                                            "car", "small_business", "other"])
    go = st.form_submit_button("Score applicant")

if go:
    payload = dict(loan_amnt=loan_amnt, term_months=term_months, installment=installment, annual_inc=annual_inc,
                   dti=dti, revol_util=revol_util, revol_bal=revol_bal, delinq_2yrs=delinq, open_acc=open_acc,
                   pub_rec=pub_rec, total_acc=total_acc, fico_range_low=fico, fico_range_high=fico + 4,
                   emp_years=emp_years, credit_history_years=history, home_ownership=home,
                   verification_status=verif, purpose=purpose)
    try:
        r = requests.post(f"{api}/score", json=payload, timeout=10)
        r.raise_for_status()
        res = r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not get a score from {api}. Is the API running? ({type(e).__name__})")
    else:
        if res.get("warning"):
            st.warning(res["warning"])
        d = res["decision"]
        {"approve": st.success, "review": st.warning, "decline": st.error}[d](f"Suggested decision: {d.upper()}")
        c1, c2 = st.columns(2)
        c1.metric("Default probability", f"{res['default_probability']:.1%}")
        c2.metric("Risk band", res["risk_band"])
        st.subheader("Main reasons for this outcome")
        for reason in res["top_risk_factors"] or ["None: no adverse factors to report."]:
            st.write(f"- {reason}")
        st.caption(f"Decline cutoff: {res['cutoff']:.1%} | model: {res['model']} | {res['disclaimer']}")
