import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from fpdf import FPDF
import io

# Function to clean a dataset
def clean_dataset(df):
    # Supprimer les colonnes 'Code', 'C.N.I.', 'pause' et toutes les colonnes commençant par 'Unnamed:'
    colonnes_a_supprimer = ['Code', 'C.N.I.', 'pause'] + [col for col in df.columns if 'Unnamed' in col]
    df.drop(columns=colonnes_a_supprimer, inplace=True)

    # Remplacer les valeurs invalides par NaN et convertir en type float
    df['Total présence'] = pd.to_numeric(df['Total présence'], errors='coerce')
    # Suppression des lignes où 'Nom' est vide
    df = df.dropna(subset=['Nom'])

    # Suppression des lignes où 'Horaire/Incidence' est 'Holiday'
    df = df[df['Horaire/Incidence'] != 'Holiday']
    # Remplacement des valeurs manquantes dans les colonnes 'Entrée' et 'Sortie'
    # Remplacement de 'Pointage manquant' par 0
    df.replace({'Pointage manquant': '00:00'}, inplace=True)

    # Conversion des colonnes en format datetime
    cols_to_convert = ['Entrée', 'Sortie', 'Entrée.1', 'Sortie.1', 'Entrée.2', 'Sortie.2']
    df[cols_to_convert] = df[cols_to_convert].apply(pd.to_datetime, format='%H:%M', errors='coerce')

    # Conversion de la colonne 'Date'
    df['Date'] = pd.to_datetime(df['Date'].str.split(' ').str[0], format='%d/%m/%Y')

    # Extraction des premières et dernières heures d'entrée/sortie
    df['Entrée'] = df[['Entrée', 'Entrée.1', 'Entrée.2']].min(axis=1)
    df['Sortie'] = df[['Sortie', 'Sortie.1', 'Sortie.2']].max(axis=1)

    # Suppression des colonnes supplémentaires
    df = df.drop(columns=['Entrée.1', 'Sortie.1', 'Entrée.2', 'Sortie.2'])
    # Calcul du nombre d'heures passées au bureau
    df['Total présence'] = (df['Sortie'] - df['Entrée']).dt.total_seconds() / 3600
    df['Total présence'].fillna(0, inplace=True)  # Remplace les valeurs manquantes par 0
    df = df.loc[df['SERVICE:'] != 'Cabinet']
    return df

# Function to merge datasets
def merge_datasets(dfs):
    return pd.concat(dfs, ignore_index=True)

# Function to process datasets and perform additional operations
def process_data(df):
    # Add your additional logic here
    df['Type'] = 'retard'
    absence = 0
    normal = 8
    df.loc[df['Total présence'] == absence, 'Type'] = 'absence'
    df.loc[df['Total présence'] < absence, 'Type'] = 'Pointage manquant'
    df.loc[df['Total présence'] >= normal, 'Type'] = 'normal'
    df['Type'] = np.where(df['Entrée'].dt.hour > 8, 'retard', df['Type'])
    df['Type'] = np.where(df['Sortie'].dt.hour < 17, 'partie tôt', df['Type'])
    
    # Example of removing specific names
    noms_a_supprimer = [
    "Amadou Legrand DIOP", "Pape Waly DIOUF", "Mamadou Nana SARR", "Diegane DIAGNE",
    "El Hadji Medoune DIOUF", "Ndéye Bineta NDIAYE", "Amath SALL", "Aminata SOW", 
    "Thioro MBAYE SALL", "Madogal THIOUNE", "issa CISSE", "Yankhoba NDIAYE", 
    "Aboune DIATTA", "Yaye Nogueye KEITA", "Ibrahima DIOUF", "Louis Benoit MBAYE", 
    "Ibrahima NDIAYE", "Ndeye fatou FALL", "Mbaye DIENG", "Serigne Mansour FAYE"]
    noms_a_supprimer = [nom.lower() for nom in noms_a_supprimer]
    df['Nom_lower'] = df['Nom'].str.lower()
    df = df[~df['Nom_lower'].isin(noms_a_supprimer)]
    df.drop(columns=['Nom_lower'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df


def perform_analysis(df, save_plots=False):
    """
    Function to perform data analysis and generate visual reports.
    If save_plots is True, it saves the plots to files for PDF inclusion.
    If save_plots is False, it displays the plots in Streamlit.
    """

    # Dictionnaire pour stocker les chemins des fichiers d'image
    plot_paths = {}

    # Introduction
    if not save_plots:
        st.markdown("## Introduction")
        st.markdown("Ce rapport presente une analyse des donnees de pointage des employes, en se concentrant sur les absences et les retards par service et par direction.")

    # Analyse des Absences
    if not save_plots:
        st.markdown("## Analyse des Absences")
        st.write("Cette section présente les résultats de l'analyse des absences par service et par direction.")

    # Répartition Globale des Types de Journée
    if not save_plots:
        st.markdown("## Répartition Globale des Types de Journée")
    fig, ax = plt.subplots()
    df['Type'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title("Distribution des Types de Journée")

    if save_plots:
        plot_path = './datas/images/type_distribution.png'
        fig.tight_layout()  # Ajuster les marges automatiquement
        fig.savefig(plot_path, bbox_inches='tight', dpi=300)  # Ajouter bbox_inches et dpi
       
        plot_paths['type_distribution'] = plot_path
        plt.close(fig)
    else:
        st.pyplot(fig)

    # Analyse des Retards
    if not save_plots:
        st.markdown("## Analyse des Retards")
    df_grouped = df.groupby('Date').agg({'Total présence': 'sum'})
    fig, ax = plt.subplots()
    df_grouped.plot(kind='line', ax=ax)
    ax.set_title("Total Présence par Jour")

    if save_plots:
        plot_path = './datas/images/total_presence_per_day.png'
        fig.savefig(plot_path, bbox_inches='tight', dpi=300)
        plot_paths['total_presence'] = plot_path
        plt.close(fig)
    else:
        st.pyplot(fig)

    # Tendance Mensuelle des Présences et Retards
    if not save_plots:
        st.markdown("## Tendance Mensuelle des Presences et Retards")
    df['Month'] = df['Date'].dt.to_period('M')
    fig, ax = plt.subplots()
    df_monthly_presence = df.groupby('Month')['Total présence'].sum()
    df_monthly_presence.plot(kind='line', ax=ax)
    ax.set_title("Tendance Mensuelle des Présences")

    if save_plots:
        plot_path = './datas/images/monthly_presence_trend.png'
        fig.savefig(plot_path, bbox_inches='tight', dpi=300)
        plot_paths['monthly_presence_trend'] = plot_path
        plt.close(fig)
    else:
        st.pyplot(fig)

    # Analyse Globale des Types de Journée
    if not save_plots:
        st.markdown("## Analyse Globale des Types de Journee")
    plt.figure(figsize=(10, 6))
    type_counts = df['Type'].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=type_counts.index, y=type_counts.values, palette='Blues_d', ax=ax)
    ax.set_title("Répartition des Types de Journée")
    plt.xlabel("Type de Journée")
    plt.ylabel("Nombre d'Employés")

    for i in range(len(type_counts)):
        ax.text(i, type_counts.values[i] + 50, str(type_counts.values[i]), ha='center')

    if save_plots:
        plot_path = './datas/images/type_global_analysis.png'
        fig.savefig(plot_path, bbox_inches='tight', dpi=300)
        plot_paths['type_global_analysis'] = plot_path
        plt.close(fig)
    else:
        st.pyplot(fig)

    # Top 10 Employés par Direction (Absences)
    if not save_plots:
        st.markdown("## Top 10 par Direction (Les Employes avec le Plus d'Absences)")
    directions = df['DIRECTION:'].unique()
    
    for direction in directions:
        absences_per_employee = df[(df['DIRECTION:'] == direction) & (df['Type'] == 'absence')].groupby('Nom').size()
        top_10_absences = absences_per_employee.nlargest(10)

        fig, ax = plt.subplots(figsize=(10, 6))
        top_10_absences.plot(kind='bar', color='orange', ax=ax)
        ax.set_title(f"Top 10 des Employes avec le Plus d'Absences ({direction})")
        plt.xlabel("Employé")
        plt.ylabel("Nombre d'Absences")

        for idx, value in enumerate(top_10_absences):
            ax.text(idx, value, int(value), ha='center', va='bottom', fontsize=12)

        if save_plots:
            plot_path = f'./datas/images/absences_{direction}.png'
            fig.savefig(plot_path, bbox_inches='tight', dpi=300)
            plot_paths[f'absences_{direction}'] = plot_path
            plt.close(fig)
        else:
            st.pyplot(fig)

    # Top 10 Employés par Direction (Retards)
    if not save_plots:
        st.markdown("## Top 10 par Direction (Les Employes avec le Plus de Retards)")
    
    for direction in directions:
        retards_per_employee = df[(df['DIRECTION:'] == direction) & (df['Type'] == 'retard')].groupby('Nom').size()
        top_10_retards = retards_per_employee.nlargest(10)

        fig, ax = plt.subplots(figsize=(10, 6))
        top_10_retards.plot(kind='bar', color='red', ax=ax)
        ax.set_title(f"Top 10 des Employés avec le Plus de Retards ({direction})")
        plt.xlabel("Employé")
        plt.ylabel("Nombre de Retards")

        for idx, value in enumerate(top_10_retards):
            ax.text(idx, value, int(value), ha='center', va='bottom', fontsize=12)

        if save_plots:
            plot_path = f'./datas/images/retards_{direction}.png'
            fig.savefig(plot_path, bbox_inches='tight', dpi=300)
            plot_paths[f'retards_{direction}'] = plot_path
            plt.close(fig)
        else:
            st.pyplot(fig)

    # Liste des Employés Pointant Après 8h
    if not save_plots:
        st.markdown("## Liste des Employes Pointant Après 8h")
        late_employees = df[df['Entrée'].dt.hour > 8]['Nom'].unique()
        st.write(late_employees)

    # Classement Graphique des Services avec le Plus d'Absences
    if not save_plots:
        st.markdown("## Classement Graphique des Services avec le Plus d'Absences")
    fig, ax = plt.subplots()
    absences_by_service = df[df['Type'] == 'absence'].groupby('SERVICE:')['Total présence'].count().sort_values(ascending=False)
    absences_by_service.plot(kind='bar', ax=ax)
    ax.set_title("Services avec le Plus d'Absences")

    if save_plots:
        plot_path = './datas/images/absences_by_service.png'
        fig.savefig(plot_path, bbox_inches='tight', dpi=300)
        plot_paths['absences_by_service'] = plot_path
        plt.close(fig)
    else:
        st.pyplot(fig)

    # Tendance Mensuelle des Absences
    if not save_plots:
        st.markdown("## Tendance Mensuelle des Absences")
    absences_per_month = df[df['Type'] == 'absence'].groupby(df['Date'].dt.to_period('M'))['Total présence'].count()
    fig, ax = plt.subplots()
    absences_per_month.plot(kind='line', ax=ax)
    ax.set_title("Absences par Mois")

    if save_plots:
        plot_path = './datas/images/absences_per_month.png'
        fig.savefig(plot_path, bbox_inches='tight', dpi=300)
        plot_paths['absences_per_month'] = plot_path
        plt.close(fig)
    else:
        st.pyplot(fig)

    # Classement Graphique des Directions avec le Plus d'Absences
    if not save_plots:
        st.markdown("## Classement Graphique des Directions avec le Plus d\'Absences")
    fig, ax = plt.subplots()
    absences_by_direction = df[df['Type'] == 'absence'].groupby('DIRECTION:')['Total présence'].count().sort_values(ascending=False)
    absences_by_direction.plot(kind='bar', ax=ax)
    ax.set_title("Directions avec le Plus d'Absences")

    if save_plots:
        plot_path = './datas/images/absences_by_direction.png'
        fig.savefig(plot_path, bbox_inches='tight', dpi=300)
        plot_paths['absences_by_direction'] = plot_path
        plt.close(fig)
    else:
        st.pyplot(fig)

    # Analyse par Heure d'Entrée et de Sortie
    if not save_plots:
        st.markdown("## Analyse par Heure d\'Entree et de Sortie")
    fig, ax = plt.subplots()
    sns.kdeplot(df['Entrée'].dt.hour.dropna(), label='Entrée', ax=ax)
    sns.kdeplot(df['Sortie'].dt.hour.dropna(), label='Sortie', ax=ax)
    ax.set_title("Distribution des Heures d'Entrée et de Sortie")
    ax.legend()

    if save_plots:
        plot_path = './datas/images/entrance_exit_distribution.png'
        fig.savefig(plot_path, bbox_inches='tight', dpi=300)
        plot_paths['entrance_exit_distribution'] = plot_path
        plt.close(fig)
    else:
        st.pyplot(fig)

    # Conclusion
    if not save_plots:
        st.markdown("## Conclusion")
        st.markdown("""
        Cette analyse détaillée du dataset de présence des employés permet de mettre en lumière plusieurs aspects importants :

        - Les types d'incidents dominants, comme les retards ou les absences.
        - L'évolution quotidienne de la présence des employés, qui peut montrer des tendances utiles pour ajuster les horaires de travail.
        - Les employés les plus présents globalement et au sein de chaque direction, offrant un aperçu des comportements exemplaires.
        - Les retards et les absences, analysés en détail pour cibler des jours ou des périodes spécifiques nécessitant des interventions.

        Ces analyses peuvent guider des décisions managériales éclairées, comme l'ajustement des horaires, la mise en place de mesures incitatives, ou le renforcement des politiques de ponctualité.

        """)

    # Return plot paths if saving plots
    if save_plots:
        return plot_paths


# Function to export the same analysis (from perform_analysis) as a PDF report

def generate_pdf_report(df):
    """
    Function to generate a PDF report by calling the perform_analysis function 
    and embedding the saved plots in the PDF. No text or titles will be shown on screen.
    """

    # Create the PDF object
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title for the PDF report
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(200, 10, txt="Rapport d'Analyse des Données de Pointage", ln=True, align='C')

    # Introduction in the PDF
    pdf.set_font('Arial', '', 12)
    pdf.ln(10)
    pdf.multi_cell(200, 10, txt="Ce rapport présente une analyse des données de pointage des employés, "
                                "en se concentrant sur les absences et les retards par service et par direction.")
    
    # Perform the analysis and save the plots
    plot_paths = perform_analysis(df, save_plots=True)

    # Add a section title for "Répartition Globale des Types de Journée" in the PDF
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(200, 10, txt="Répartition Globale des Types de Journée", ln=True, align='L')
    pdf.ln(10)
    
    # Add the corresponding chart to the PDF
    if 'type_distribution' in plot_paths:
        pdf.image(plot_paths['type_distribution'], x=10, y=None, w=150)
        pdf.ln(10)

    # Add a section for "Analyse des Retards" in the PDF
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(200, 10, txt="Analyse des Retards", ln=True, align='L')
    pdf.ln(10)

    # Add the corresponding chart
    if 'total_presence' in plot_paths:
        pdf.image(plot_paths['total_presence'], x=10, y=None, w=150)
        pdf.ln(10)

    # Add a section for "Tendance Mensuelle des Présences et Retards"
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(200, 10, txt="Tendance Mensuelle des Présences et Retards", ln=True, align='L')
    pdf.ln(10)

    if 'monthly_presence_trend' in plot_paths:
        pdf.image(plot_paths['monthly_presence_trend'], x=10, y=None, w=150)
        pdf.ln(10)

    # Add "Top 10 Employés par Direction (Absences)"
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(200, 10, txt="Top 10 Employés par Direction (Absences)", ln=True, align='L')
    pdf.ln(10)

    for direction in df['DIRECTION:'].unique():
        direction_key = f'absences_{direction}'
        if direction_key in plot_paths:
            pdf.add_page()
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(200, 10, txt=f"Top 10 Absences - {direction}", ln=True, align='L')
            pdf.image(plot_paths[direction_key], x=10, y=None, w=150)
            pdf.ln(10)

    # Add "Top 10 Employés par Direction (Retards)"
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(200, 10, txt="Top 10 Employés par Direction (Retards)", ln=True, align='L')
    pdf.ln(10)

    for direction in df['DIRECTION:'].unique():
        direction_key = f'retards_{direction}'
        if direction_key in plot_paths:
            pdf.add_page()
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(200, 10, txt=f"Top 10 Retards - {direction}", ln=True, align='L')
            pdf.image(plot_paths[direction_key], x=10, y=None, w=150)
            pdf.ln(10)

    # Add "Classement Graphique des Services avec le Plus d'Absences"
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(200, 10, txt="Classement Graphique des Services avec le Plus d'Absences", ln=True, align='L')
    pdf.ln(10)

    if 'absences_by_service' in plot_paths:
        pdf.image(plot_paths['absences_by_service'], x=10, y=None, w=150)
        pdf.ln(10)

    # Add "Tendance Mensuelle des Absences"
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(200, 10, txt="Tendance Mensuelle des Absences", ln=True, align='L')
    pdf.ln(10)

    if 'absences_per_month' in plot_paths:
        pdf.image(plot_paths['absences_per_month'], x=10, y=None, w=150)
        pdf.ln(10)

    # Add "Classement Graphique des Directions avec le Plus d'Absences"
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(200, 10, txt="Classement Graphique des Directions avec le Plus d'Absences", ln=True, align='L')
    pdf.ln(10)

    if 'absences_by_direction' in plot_paths:
        pdf.image(plot_paths['absences_by_direction'], x=10, y=None, w=150)
        pdf.ln(10)

    # Add "Analyse par Heure d'Entrée et de Sortie"
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(200, 10, txt="Analyse par Heure d'Entrée et de Sortie", ln=True, align='L')
    pdf.ln(10)

    if 'entrance_exit_distribution' in plot_paths:
        pdf.image(plot_paths['entrance_exit_distribution'], x=10, y=None, w=150)

      # Add "Conclusion"
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(200, 10, txt="Conclusion", ln=True, align='L')
    pdf.ln(10)

    # Conclusion content
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(200, 10, txt=(
        "Cette analyse détaillée du dataset de présence des employés permet de mettre en lumière plusieurs aspects importants :\n\n"
        "- Les types d'incidents dominants, comme les retards ou les absences.\n"
        "- L'évolution quotidienne de la présence des employés, qui peut montrer des tendances utiles pour ajuster les horaires de travail.\n"
        "- Les employés les plus présents globalement et au sein de chaque direction, offrant un aperçu des comportements exemplaires.\n"
        "- Les retards et les absences, analysés en détail pour cibler des jours ou des périodes spécifiques nécessitant des interventions.\n\n"
        "Ces analyses peuvent guider des décisions managériales éclairées, comme l'ajustement des horaires, la mise en place de mesures incitatives, "
        "ou le renforcement des politiques de ponctualité.\n\n"
    ))
    # Export the PDF to a byte stream
    pdf_output = io.BytesIO()
    pdf_output.write(pdf.output(dest='S').encode('latin1'))
    pdf_output.seek(0)
    
    return pdf_output

# Streamlit Interface

def main():
    st.title("Rapport d'Analyse des Données de Pointage")

    # Upload multiple datasets (csv, xls, xlsx)
    uploaded_files = st.file_uploader(
        "Televersez vos données (csv, xls, xlsx)", 
        accept_multiple_files=True, 
        type=['csv', 'xlsx', 'xls']
    )

    if uploaded_files:
        dataframes = []
        css_content = ""

        for file in uploaded_files:
            # Handle CSV files
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
                dataframes.append(df)
            
            # Handle Excel files (XLS, XLSX)
            elif file.name.endswith('.xls'):
                # Use 'xlrd' engine for .xls files
                df = pd.read_excel(file, engine='xlrd')
                
                # Convert Excel data to CSV
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)
                
                # Read CSV content for further processing
                df = pd.read_csv(csv_buffer)
                dataframes.append(df)
            
            elif file.name.endswith('.xlsx'):
                # Use 'openpyxl' engine for .xlsx files
                df = pd.read_excel(file, engine='openpyxl')
                
                # Convert Excel data to CSV
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)
                
                # Read CSV content for further processing
                df = pd.read_csv(csv_buffer)
                dataframes.append(df)

        # Proceed if at least one data file was uploaded
        if dataframes:
            # Merge datasets
            merged_df = merge_datasets(dataframes)

            # Clean datasets
            cleaned_df = clean_dataset(merged_df)
            
            # Process data
            processed_df = process_data(cleaned_df)

            # Perform analysis (like in the notebook)
            st.header("Rapport d'Analyse")
            perform_analysis(processed_df)

            # Export to PDF
            if st.button("Générer le rapport en PDF"):
                pdf = generate_pdf_report(processed_df)
                st.download_button(label="Télécharger le rapport", data=pdf, file_name="report.pdf")

# Run the Streamlit app
if __name__ == '__main__':
    main()
