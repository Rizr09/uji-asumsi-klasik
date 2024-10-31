import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Set up page config
st.set_page_config(
    page_title="Linear Regression Assumptions Dashboard",
    page_icon="📊",
    layout="wide",
)

# Shadcn theme settings and custom CSS
st.markdown(
    """
    <style>
    .main { background-color: #f5f5f5; }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px;
    }
    .stAlert > div {
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .test-definition {
        background-color: #f0f0f0;
        border-left: 5px solid #4CAF50;
        padding: 1rem;
        margin-top: 0.5rem;
    }
    .interpretation {
        background-color: #e6f3ff;
        border-left: 5px solid #007bff;
        padding: 1.5rem;  /* Tambah padding */
        margin-top: 0.5rem;
        margin-bottom: 1rem;  /* Beri jarak ke bawah */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and description
st.title("📊 Enhanced Linear Regression Assumptions Dashboard")
st.write("""
This dashboard allows you to test and visualize the assumptions of a linear regression model.
Upload one or multiple datasets, select the dependent variable (target), and inspect the assumptions.
""")

# Sidebar for dataset uploads and options
st.sidebar.header("Upload Your Datasets")
uploaded_files = st.sidebar.file_uploader(
    "Upload Excel files", accept_multiple_files=True, type=["xlsx", "xls"])

# Function to display test definition, formula, and interpretation


def show_test_info(title, definition, formula, interpretation):
    with st.expander(f"i {title} Info"):
        st.markdown(f"**Definition:** {definition}")
        st.markdown(f"**Formula:**")
        st.latex(formula)
        st.markdown("**Interpretation:**")
        st.markdown(
            f'<div class="interpretation">{interpretation}</div>', unsafe_allow_html=True
        )


# Function to perform linear regression and assumption tests
def perform_regression_analysis(X, y):
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    # Linearity test (Residuals vs Fitted Values Plot)
    fig1 = px.scatter(
        x=model.fittedvalues,
        y=model.resid,
        labels={'x': 'Fitted values', 'y': 'Residuals'},
        title='Residuals vs Fitted Values'
    )
    fig1.add_shape(type="line", x0=min(model.fittedvalues), y0=0, x1=max(
        model.fittedvalues), y1=0, line=dict(color="red", dash="dash"))

    # Independence test (Durbin-Watson)
    dw_stat = durbin_watson(model.resid)

    # Homoscedasticity test (Breusch-Pagan)
    _, pval, __, f_pval = het_breuschpagan(model.resid, model.model.exog)

    # Normality test (Shapiro-Wilk Test)
    shapiro_test = stats.shapiro(model.resid)

    # Q-Q Plot for Normality Test
    qq_fig = sm.qqplot(model.resid, line="45", fit=True)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=qq_fig.gca().lines[1].get_xdata(
    ), y=qq_fig.gca().lines[1].get_ydata(), mode="markers", name="Residuals"))
    fig2.add_trace(go.Scatter(x=qq_fig.gca().lines[0].get_xdata(), y=qq_fig.gca(
    ).lines[0].get_ydata(), mode="lines", name="45-degree line"))
    fig2.update_layout(title="Q-Q Plot (Normality Test)",
                       xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles")

    # Multicollinearity test (VIF)
    vif_data = pd.DataFrame()
    vif_data["VIF"] = [variance_inflation_factor(
        X.values, i) for i in range(X.shape[1])]
    vif_data["Feature"] = X.columns
    vif_data = vif_data[vif_data["Feature"] != "const"]

    return model, fig1, dw_stat, pval, shapiro_test, fig2, vif_data

# New function to handle violations and apply solutions
def handle_violations(X, y, violation_type):
    X = sm.add_constant(X)
    if violation_type == "heteroscedasticity":
        # Apply weighted least squares
        model = sm.OLS(y, X).fit()
        weights = 1 / (model.resid**2)
        weighted_model = sm.WLS(y, X, weights=weights).fit()
        return weighted_model, X, y
    elif violation_type == "non_normality":
        # Apply Box-Cox transformation
        y_transformed, lambda_param = stats.boxcox(y)
        transformed_model = sm.OLS(y_transformed, X).fit()
        return transformed_model, X, y_transformed
    elif violation_type == "autocorrelation":
        # Apply first-order differencing
        y_diff = y.diff().dropna()
        X_diff = X.diff().dropna()
        diff_model = sm.OLS(y_diff, X_diff).fit()
        return diff_model, X_diff, y_diff

# Fungsi untuk menghasilkan laporan diagnostik
def generate_diagnostic_report(dw_stat, pval_bp, shapiro_pval, vif_data):
    report = ""

    # Linearitas
    report += "## 1. Linearitas\n"
    report += "Linearitas harus diperiksa secara visual menggunakan plot Residual vs Nilai Prediksi.\n\n"

    # Independensi
    report += "## 2. Independensi (Durbin-Watson)\n"
    if 1.5 < dw_stat < 2.5:
        report += f"Statistik Durbin-Watson: {dw_stat:.2f}\n"
        report += "\nInterpretasi: Residual kemungkinan besar independen.\n"
    else:
        report += f"Statistik Durbin-Watson: {dw_stat:.2f}\n"
        report += "\nInterpretasi: Kemungkinan ada autokorelasi dalam residual.\n"
        report += "\nSolusi:\n"
        report += "- Pertimbangkan untuk menambahkan variabel lag atau differencing pada data time series.\n"
        report += "- Gunakan model regresi yang robust terhadap autokorelasi, seperti Generalized Least Squares (GLS).\n\n"

    # Homoskedastisitas
    report += "## 3. Homoskedastisitas (Breusch-Pagan)\n"
    if pval_bp > 0.05:
        report += f"Nilai p Breusch-Pagan: {pval_bp:.4f}\n"
        report += "\nInterpretasi: Residual kemungkinan memiliki varians konstan (homoskedastis).\n"
    else:
        report += f"Nilai p Breusch-Pagan: {pval_bp:.4f}\n"
        report += "\nInterpretasi: Residual mungkin memiliki varians tidak konstan (heteroskedastis).\n"
        report += "\nSolusi:\n"
        report += "- Gunakan transformasi pada variabel (misalnya, log transformation).\n"
        report += "- Gunakan Weighted Least Squares (WLS) atau robust standard errors.\n"
        report += "- Pertimbangkan untuk menggunakan model heteroskedastisitas lainnya seperti GARCH untuk data time series.\n\n"

    # Normalitas
    report += "## 4. Normalitas (Shapiro-Wilk)\n"
    if shapiro_pval > 0.05:
        report += f"Nilai p Shapiro-Wilk: {shapiro_pval:.4f}\n"
        report += "\nInterpretasi: Residual kemungkinan terdistribusi normal.\n"
    else:
        report += f"Nilai p Shapiro-Wilk: {shapiro_pval:.4f}\n"
        report += "\nInterpretasi: Residual mungkin tidak terdistribusi normal.\n"
        report += "\nSolusi:\n"
        report += "- Periksa dan tangani outlier jika ada.\n"
        report += "- Coba transformasi data (misalnya, log, square root, Box-Cox).\n"
        report += "- Jika ukuran sampel besar, pertimbangkan untuk menggunakan metode robust yang tidak mengasumsikan normalitas.\n\n"

    # Multikolinearitas
    report += "## 5. Multikolinearitas (VIF)\n"
    max_vif = vif_data["VIF"].max()
    if max_vif > 5:
        report += f"VIF maksimum: {max_vif:.2f}\n"
        report += "\nInterpretasi: Beberapa fitur mungkin memiliki multikolinearitas tinggi.\n"
        report += "\nSolusi:\n"
        report += "- Hapus salah satu dari variabel yang sangat berkorelasi.\n"
        report += "- Gunakan metode reduksi dimensi seperti PCA.\n"
        report += "- Pertimbangkan untuk menggunakan teknik regularisasi seperti Ridge atau Lasso regression.\n"
    else:
        report += f"VIF maksimum: {max_vif:.2f}\n"
        report += "\nInterpretasi: Tidak terdeteksi multikolinearitas yang signifikan.\n"

    return report


# Main application logic
if uploaded_files:
    # Process uploaded datasets
    try:
        datasets = {file.name: pd.read_excel(file) for file in uploaded_files}
    except ValueError as e:
        st.error(f"Error reading one of the files: {e}")

    # Dataset selection
    selected_dataset = st.sidebar.selectbox(
        "Choose a dataset", options=list(datasets.keys()))
    data = datasets[selected_dataset]

    # Show preview of the selected dataset
    st.write(f"### Dataset: {selected_dataset}")
    st.write(data.head())

    # Variable selection
    dependent_var = st.sidebar.selectbox(
        "Select the dependent variable", options=data.columns)
    independent_vars = st.sidebar.multiselect("Select independent variables", options=[
                                              col for col in data.columns if col != dependent_var])

    if dependent_var and independent_vars:
        # Prepare the data for linear regression
        X = data[independent_vars]
        y = data[dependent_var]

        # Perform regression analysis on the entire dataset
        model, fig1, dw_stat, bp_pvalue, shapiro_test, fig2, vif_data = perform_regression_analysis(X, y)

        # Display initial model summary and test results
        st.subheader("Initial Model Summary")
        st.text(model.summary())

        # Check for violations and offer solutions
        violations = []
        if bp_pvalue <= 0.05:
            violations.append("heteroscedasticity")
        if shapiro_test.pvalue <= 0.05:
            violations.append("non_normality")
        if dw_stat < 1.5 or dw_stat > 2.5:
            violations.append("autocorrelation")

        if violations:
            st.warning("The following assumptions are violated:")
            for violation in violations:
                st.write(f"- {violation.capitalize()}")

            selected_violation = st.selectbox("Choose a violation to address:", violations)
            if st.button("Apply Solution"):
                new_model, new_X, new_y = handle_violations(X, y, selected_violation)
                
                # Perform regression analysis on the new model
                new_model, new_fig1, new_dw_stat, new_bp_pvalue, new_shapiro_test, new_fig2, new_vif_data = perform_regression_analysis(new_X, new_y)

                st.subheader("Comparison of Models")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Original Model")
                    st.metric("R-squared", f"{model.rsquared:.4f}")
                    st.metric("Adjusted R-squared", f"{model.rsquared_adj:.4f}")
                    st.metric("AIC", f"{model.aic:.2f}")
                    st.metric("Durbin-Watson", f"{dw_stat:.4f}")
                    st.metric("Breusch-Pagan p-value", f"{bp_pvalue:.4f}")
                    st.metric("Shapiro-Wilk p-value", f"{shapiro_test.pvalue:.4f}")
                    st.plotly_chart(fig1, use_container_width=True, key="original_fig1")
                    st.plotly_chart(fig2, use_container_width=True, key="original_fig2")
                with col2:
                    st.write("Improved Model")
                    st.metric("R-squared", f"{new_model.rsquared:.4f}")
                    st.metric("Adjusted R-squared", f"{new_model.rsquared_adj:.4f}")
                    st.metric("AIC", f"{new_model.aic:.2f}")
                    st.metric("Durbin-Watson", f"{new_dw_stat:.4f}")
                    st.metric("Breusch-Pagan p-value", f"{new_bp_pvalue:.4f}")
                    st.metric("Shapiro-Wilk p-value", f"{new_shapiro_test.pvalue:.4f}")
                    st.plotly_chart(new_fig1, use_container_width=True, key="new_fig1")
                    st.plotly_chart(new_fig2, use_container_width=True, key="new_fig2")

                st.subheader("Improved Model Summary")
                st.text(new_model.summary())

                # Display updated VIF data
                st.subheader("Updated Multicollinearity Test (VIF)")
                st.write(new_vif_data)


        # 1. Linearity test
        st.subheader("1. Uji Linearitas")
        show_test_info(
            "Linearitas",
            "Harus ada hubungan linear antara variabel independen dan variabel dependen.",
            r"y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n + \epsilon",
            "Periksa apakah residual tersebar secara acak di sekitar garis horizontal pada 0."
        )
        st.plotly_chart(fig1, key="linearity_test")
        st.info(
            "Interpretasikan plot: Lihat penyebaran acak di sekitar garis horizontal pada 0.")


        # 2. Independence test
        st.subheader("2. Uji Independensi (Statistik Durbin-Watson)")
        show_test_info(
            "Uji Durbin-Watson",
            "Memeriksa autokorelasi dalam residual.",
            r"DW = \frac{\sum_{t=2}^{n} (e_t - e_{t-1})^2}{\sum_{t=1}^{n} e_t^2}",
            """
            DW ≈ 2: Tidak ada autokorelasi.
            DW < 1.5 atau DW > 2.5: Kemungkinan ada autokorelasi.
            """
        )
        st.metric("Statistik Durbin-Watson", f"{dw_stat:.2f}")
        if 1.5 < dw_stat < 2.5:
            st.success("Residual kemungkinan besar independen.")
        else:
            st.warning("Kemungkinan ada autokorelasi dalam residual.")

        # 3. Homoscedasticity test
        st.subheader("3. Uji Homoskedastisitas (Breusch-Pagan)")
        show_test_info(
            "Uji Breusch-Pagan",
            "Menguji varians konstan dalam residual (homoskedastisitas).",
            r"BP = nR^2 \sim \chi^2_{(p-1)}",
            "Jika p > 0.05, residual kemungkinan memiliki varians konstan."
        )
        st.metric("Nilai p Breusch-Pagan", f"{bp_pvalue:.4f}")
        if bp_pvalue > 0.05:
            st.success("Residual kemungkinan memiliki varians konstan.")
        else:
            st.warning("Residual mungkin memiliki varians tidak konstan.")

        # 4. Normality test
        st.subheader("4. Uji Normalitas (Uji Shapiro-Wilk)")
        show_test_info(
            "Uji Shapiro-Wilk",
            "Menguji apakah residual terdistribusi normal.",
            r"W = \frac{(\sum_{i=1}^n a_i x_{(i)})^2}{\sum_{i=1}^n (x_i - \bar{x})^2}",
            "Jika p > 0.05, residual kemungkinan terdistribusi normal."
        )

        st.metric("Nilai p Shapiro-Wilk", f"{shapiro_test.pvalue:.4f}")
        if shapiro_test.pvalue > 0.05:
            st.success("Residual kemungkinan terdistribusi normal.")
        else:
            st.warning("Residual mungkin tidak terdistribusi normal.")

        # Q-Q plot for normality
        st.plotly_chart(fig2, key="qq_plot")

        # 5. Multicollinearity test (Variance Inflation Factor - VIF)
        st.subheader(
            "5. Uji Multikolinearitas (Variance Inflation Factor - VIF)")
        show_test_info(
            "Variance Inflation Factor",
            "Mengukur tingkat multikolinearitas dalam model regresi.",
            r"VIF = \frac{1}{1 - R_j^2}",
            """
            - **VIF ≈ 1**: Tidak ada multikolinearitas.  
            - **1 < VIF < 10**: Multikolinearitas sedang.  
            - **VIF > 10**: Multikolinearitas tinggi.
            """
        )
        st.write(vif_data)

        # Provide insights on multicollinearity
        if vif_data["VIF"].max() > 10:
            st.warning(
                "Beberapa fitur mungkin memiliki multikolinearitas tinggi (VIF > 10). Pertimbangkan untuk menghapus atau menggabungkan variabel yang sangat berkorelasi.")
        else:
            st.success(
                "Tidak terdeteksi multikolinearitas yang signifikan (VIF < 10).")

        # Pindahkan Laporan Hasil Uji Diagnostik ke bagian paling bawah
        st.subheader("Laporan Hasil Uji Diagnostik")
        diagnostic_report = generate_diagnostic_report(
            dw_stat, bp_pvalue, shapiro_test.pvalue, vif_data)
        st.markdown(diagnostic_report)

        # # Opsi untuk mengunduh laporan diagnostik
        # st.download_button(
        #     label="Unduh Laporan Diagnostik",
        #     data=diagnostic_report,
        #     file_name="laporan_diagnostik_regresi.md",
        #     mime="text/markdown"
        # )

        # # Option to download the regression model summary
        # st.download_button(
        #     label="Unduh Ringkasan Model",
        #     data=model.summary().as_text(),
        #     file_name="ringkasan_model_regresi.txt",
        #     mime="text/plain"
        # )
    else:
        st.warning("Silakan pilih variabel dependen dan independen.")
else:
    st.info("Unggah dataset untuk memulai.")
