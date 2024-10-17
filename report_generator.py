from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Fungsi untuk membuat chart residuals
def plot_residuals(model, filename="residuals.png"):
    residuals = model.resid
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, kde=True, bins=30)
    plt.title("Distribusi Residu")
    plt.savefig(filename)
    plt.close()

def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i) for i in range(X.shape[1])
    ]
    return vif_data

# Kelas PDF untuk menyusun laporan
class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "Laporan Diagnostik dan Ringkasan Model", align="C", ln=True)
        self.ln(10)

    def chapter_title(self, title):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, title, ln=True)
        self.ln(5)

    def chapter_body(self, body):
        self.set_font("Arial", "", 12)
        self.multi_cell(0, 10, body)
        self.ln()

    def add_image(self, image_path, w=150, h=100):
        self.image(image_path, x=30, w=w, h=h)
        self.ln(10)

# Fungsi untuk membuat PDF
def generate_pdf(model, diagnostic_report, filename="Laporan_Linear_Regression.pdf"):
    pdf = PDF()
    pdf.add_page()

    # Ringkasan Model
    pdf.chapter_title("Ringkasan Model")
    summary_text = model.summary().as_text()
    pdf.chapter_body(summary_text)

    # Diagnostic Report
    pdf.chapter_title("Laporan Hasil Uji Diagnostik")
    pdf.chapter_body(diagnostic_report)

    # VIF
    pdf.chapter_title("Interpretasi VIF")
    vif_data = calculate_vif(X)
    vif_text = (
        "- **VIF ≈ 1**: Tidak ada multikolinearitas.\n"
        "- **1 < VIF < 10**: Multikolinearitas sedang.\n"
        "- **VIF > 10**: Multikolinearitas tinggi."
    )
    pdf.chapter_body(vif_data.to_string(index=False))
    pdf.chapter_body(vif_text)

    # Chart Residuals
    plot_residuals(model)  # Simpan chart sebagai PNG
    pdf.chapter_title("Chart Residuals")
    pdf.add_image("residuals.png")

    # Simpan PDF
    pdf.output(filename)
    print(f"Laporan PDF '{filename}' berhasil dibuat!")
