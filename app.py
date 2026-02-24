import streamlit as st
import pandas as pd
import plotly.express as px
import os
import logging
from pathlib import Path
import tempfile
import time

# Import functions from flight_agent
from flight_agent import process_flight, DATA_ROOT, LOTS_ROOT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("flight_agent_app")

st.set_page_config(page_title="Gołąb AI - Agent Lotowy", page_icon="🐦", layout="wide")

def list_processed_flights():
    """List all processed flights found in LOTS_ROOT."""
    if not LOTS_ROOT.exists():
        return []
    flights = []
    for d in LOTS_ROOT.iterdir():
        if d.is_dir() and (d / "summary.json").exists():
            try:
                summary = pd.read_json(d / "summary.json", typ='series')
                flights.append(summary)
            except Exception as e:
                logger.error(f"Error reading summary for {d}: {e}")
    return flights

def main():
    st.title("🐦 Gołąb AI - Agent Analityczny")

    # Sidebar navigation
    mode = st.sidebar.radio("Wybierz tryb:", ["Import Danych", "Analiza Lotów", "Czat z Agentem (Beta)"])

    if mode == "Import Danych":
        render_import_page()
    elif mode == "Analiza Lotów":
        render_analysis_page()
    elif mode == "Czat z Agentem (Beta)":
        render_chat_page()

def render_import_page():
    st.header("📥 Import Danych z List Konkursowych")
    
    st.markdown("""
    Wklej linki do plików tekstowych z listami konkursowymi (np. z oddzial.com).
    Każdy link w nowej linii.
    """)

    urls_input = st.text_area("Lista URLi", height=200, help="Wklej tutaj linki, np. https://0489.oddzial.com/files/301/lotnr2_50525.txt")
    enable_weather = st.checkbox("Pobieraj pogodę (Open-Meteo)", value=False)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        start_import = st.button("🚀 Rozpocznij Import", type="primary")
    
    if start_import and urls_input:
        urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        
        for i, url in enumerate(urls):
            status_text.text(f"Przetwarzanie {i+1}/{len(urls)}: {url}")
            try:
                # Create a temporary file to download the list to
                with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
                    tmp_path = Path(tmp_file.name)
                
                # Process the flight
                # Note: We don't provide sections/lofts files here yet, assuming they exist or will be derived
                summary = process_flight(
                    list_path=tmp_path,
                    download_url=url,
                    enable_weather=enable_weather,
                    logger=logger
                )
                
                results.append({"url": url, "status": "✅ Sukces", "id": summary.get("flight_id")})
                
                # Cleanup temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                    
            except Exception as e:
                logger.error(f"Failed to process {url}: {e}")
                results.append({"url": url, "status": "❌ Błąd", "error": str(e)})
            
            progress_bar.progress((i + 1) / len(urls))
            
        status_text.text("Zakończono!")
        
        st.subheader("Wyniki Importu")
        st.dataframe(pd.DataFrame(results))

def render_analysis_page():
    st.header("📊 Analiza Lotów")
    
    flights = list_processed_flights()
    if not flights:
        st.warning("Brak przetworzonych lotów. Przejdź do zakładki 'Import Danych'.")
        return

    # Convert to DataFrame for easier display
    df_flights = pd.DataFrame(flights)
    
    # Sort by date
    if 'flight_date' in df_flights.columns:
        df_flights['flight_date'] = pd.to_datetime(df_flights['flight_date'])
        df_flights = df_flights.sort_values('flight_date', ascending=False)
    
    # Selection
    flight_options = {f"{row['flight_date'].date()} - {row['list_name']} ({row['release_point']})": row['flight_id'] for i, row in df_flights.iterrows()}
    selected_label = st.selectbox("Wybierz lot do analizy:", list(flight_options.keys()))
    
    if selected_label:
        flight_id = flight_options[selected_label]
        show_flight_details(flight_id)

def show_flight_details(flight_id):
    flight_dir = LOTS_ROOT / flight_id
    report_path = flight_dir / "report.csv"
    summary_path = flight_dir / "summary.json"
    
    if not report_path.exists():
        st.error("Brak pliku raportu dla tego lotu.")
        return

    # Load data
    df = pd.read_csv(report_path)
    summary = pd.read_json(summary_path, typ='series')

    # Display Summary Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Data", summary.get("flight_date"))
    col2.metric("Miejscowość", summary.get("release_point"))
    col3.metric("Wysłane Gołębie", summary.get("birds_sent"))
    col4.metric("Powroty", f"{summary.get('birds_returned')} ({int(summary.get('birds_returned'))/int(summary.get('birds_sent'))*100:.1f}%)")

    # Map Visualization (if coords available)
    st.subheader("🗺️ Mapa Hodowców")
    if 'lat' in summary and 'lon' in summary:
        # This requires fancier coords in the report. 
        # flight_agent.py build_report doesn't explicitly save fancier lat/lon in the CSV output unless we add it.
        # Let's check report columns from flight_agent.py...
        # It has 'hodowca', 'sekcja', 'kurs_stopnie', 'wiatr_opis', etc.
        # It does NOT seem to have lat/lon of the fancier in the final CSV based on my read.
        # Wait, let me check build_report again.
        pass
    
    # Display Data Table
    st.subheader("Wyniki Szczegółowe")
    st.dataframe(df)

    # Drift/Wind Analysis (Placeholder based on user request)
    st.subheader("💨 Analiza Wiatru i Dryfu")
    if 'wiatr_opis' in df.columns:
        fig = px.histogram(df, x='wiatr_opis', title="Rozkład wiatru dla hodowców")
        st.plotly_chart(fig)
    
    # Correlation: Section vs Return Rate
    if 'sekcja' in df.columns and 'ptaki_wyslane' in df.columns and 'ptaki_klasyfikowane' in df.columns:
        st.subheader("📈 Efektywność Sekcji")
        # Aggregate by section
        sect_stats = df.groupby('sekcja').agg({
            'ptaki_wyslane': 'sum',
            'ptaki_klasyfikowane': 'sum'
        }).reset_index()
        sect_stats['procent_powrotow'] = (sect_stats['ptaki_klasyfikowane'] / sect_stats['ptaki_wyslane'] * 100).round(1)
        
        fig2 = px.bar(sect_stats, x='sekcja', y='procent_powrotow', title="Procent powrotów wg sekcji", color='procent_powrotow')
        st.plotly_chart(fig2)

def render_chat_page():
    st.header("💬 Czat z Agentem")
    st.info("Ta funkcja jest w trakcie budowy. Tutaj będziesz mógł rozmawiać z agentem o wynikach lotów.")

if __name__ == "__main__":
    main()
