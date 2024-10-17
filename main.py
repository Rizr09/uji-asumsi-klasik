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
    "Upload CSV files", accept_multiple_files=True, type="csv")

# Function to display test definition, formula, and interpretation


def show_test_info(title, definition, formula, interpretation):
    with st.expander(f"ℹ️ {title} Info"):
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

    return model, fig1, dw_stat, pval, shapiro_test, fig2, vif_data

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
    datasets = {file.name: pd.read_csv(file) for file in uploaded_files}

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

    # Additional options
    st.sidebar.subheader("Additional Options")
    standardize = st.sidebar.checkbox("Standardize variables", value=False)
    test_size = st.sidebar.slider(
        "Test set size", min_value=0.1, max_value=0.5, value=0.2, step=0.1)

    if dependent_var and independent_vars:
        # Prepare the data for linear regression
        X = data[independent_vars]
        y = data[dependent_var]

        # Standardize if selected
        if standardize:
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42)

                        # Lakukan analisis regresi
        model, fig1, dw_stat, pval, shapiro_test, fig2, vif_data = perform_regression_analysis(
            X_train, y_train)

        # Tampilkan ringkasan model
        st.subheader("Ringkasan Model")
        st.text(model.summary())

        # 1. Linearity test
        st.subheader("1. Uji Linearitas")
        show_test_info(
            "Linearitas",
            "Harus ada hubungan linear antara variabel independen dan variabel dependen.",
            r"y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n + \epsilon",
            "Periksa apakah residual tersebar secara acak di sekitar garis horizontal pada 0."
        )
        st.plotly_chart(fig1)
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
        st.metric("Nilai p Breusch-Pagan", f"{pval:.4f}")
        if pval > 0.05:
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
        st.plotly_chart(fig2)

        # 5. Multicollinearity test (Variance Inflation Factor - VIF)
        st.subheader(
            "5. Uji Multikolinearitas (Variance Inflation Factor - VIF)")
        show_test_info(
            "Variance Inflation Factor",
            "Mengukur tingkat multikolinearitas dalam model regresi.",
            r"VIF = \frac{1}{1 - R_j^2}",
            """
            - **VIF ≈ 1**: Tidak ada multikolinearitas.  
            - **1 < VIF < 5**: Multikolinearitas sedang.  
            - **VIF > 5**: Multikolinearitas tinggi.
            """
        )
        st.write(vif_data)

        # Provide insights on multicollinearity
        if vif_data["VIF"].max() > 5:
            st.warning(
                "Beberapa fitur mungkin memiliki multikolinearitas tinggi (VIF > 5). Pertimbangkan untuk menghapus atau menggabungkan variabel yang sangat berkorelasi.")
        else:
            st.success("Tidak terdeteksi multikolinearitas yang signifikan (VIF < 5).")

        # Pindahkan Laporan Hasil Uji Diagnostik ke bagian paling bawah
        st.subheader("Laporan Hasil Uji Diagnostik")
        diagnostic_report = generate_diagnostic_report(dw_stat, pval, shapiro_test.pvalue, vif_data)
        st.markdown(diagnostic_report)

        # Opsi untuk mengunduh laporan diagnostik
        st.download_button(
            label="Unduh Laporan Diagnostik",
            data=diagnostic_report,
            file_name="laporan_diagnostik_regresi.md",
            mime="text/markdown"
        )

        # Option to download the regression model summary
        st.download_button(
            label="Unduh Ringkasan Model",
            data=model.summary().as_text(),
            file_name="ringkasan_model_regresi.txt",
            mime="text/plain"
        )
    else:
        st.warning("Silakan pilih variabel dependen dan independen.")
else:
    st.info("Unggah dataset untuk memulai.")