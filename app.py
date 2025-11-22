"""
🏆 D-GOL ULTIMATE 2025 - WEB APP 🏆
Versión Streamlit - Accesible desde cualquier navegador
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize
import requests
import pickle
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

st.set_page_config(
    page_title="D-GOL Ultimate 2025",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

CACHE_HORAS = 4
MIN_PARTIDOS = 15

# 38 LIGAS COMPLETAS (las mismas de tu código)
LIGAS_URLS = {
    'Premier League (ENG)': 'https://www.football-data.co.uk/mmz4281/2526/E0.csv',
    'Championship (ENG)': 'https://www.football-data.co.uk/mmz4281/2526/E1.csv',
    'League One (ENG)': 'https://www.football-data.co.uk/mmz4281/2526/E2.csv',
    'League Two (ENG)': 'https://www.football-data.co.uk/mmz4281/2526/E3.csv',
    'National League (ENG)': 'https://www.football-data.co.uk/mmz4281/2526/EC.csv',
    'Premiership (SCO)': 'https://www.football-data.co.uk/mmz4281/2526/SC0.csv',
    'Championship (SCO)': 'https://www.football-data.co.uk/mmz4281/2526/SC1.csv',
    'League One (SCO)': 'https://www.football-data.co.uk/mmz4281/2526/SC2.csv',
    'League Two (SCO)': 'https://www.football-data.co.uk/mmz4281/2526/SC3.csv',
    'La Liga (ESP)': 'https://www.football-data.co.uk/mmz4281/2526/SP1.csv',
    'Segunda División (ESP)': 'https://www.football-data.co.uk/mmz4281/2526/SP2.csv',
    'Serie A (ITA)': 'https://www.football-data.co.uk/mmz4281/2526/I1.csv',
    'Serie B (ITA)': 'https://www.football-data.co.uk/mmz4281/2526/I2.csv',
    'Bundesliga (GER)': 'https://www.football-data.co.uk/mmz4281/2526/D1.csv',
    'Bundesliga 2 (GER)': 'https://www.football-data.co.uk/mmz4281/2526/D2.csv',
    'Ligue 1 (FRA)': 'https://www.football-data.co.uk/mmz4281/2526/F1.csv',
    'Ligue 2 (FRA)': 'https://www.football-data.co.uk/mmz4281/2526/F2.csv',
    'Eredivisie (NED)': 'https://www.football-data.co.uk/mmz4281/2526/N1.csv',
    'Primeira Liga (POR)': 'https://www.football-data.co.uk/mmz4281/2526/P1.csv',
    'Jupiler Pro League (BEL)': 'https://www.football-data.co.uk/mmz4281/2526/B1.csv',
    'Süper Lig (TUR)': 'https://www.football-data.co.uk/mmz4281/2526/T1.csv',
    'Super League (GRE)': 'https://www.football-data.co.uk/mmz4281/2526/G1.csv',
    'Liga Profesional (ARG)': 'https://www.football-data.co.uk/new/ARG.csv',
    'Bundesliga (AUT)': 'https://www.football-data.co.uk/new/AUT.csv',
    'Serie A (BRA)': 'https://www.football-data.co.uk/new/BRA.csv',
    'Super League (CHN)': 'https://www.football-data.co.uk/new/CHN.csv',
    'Superliga (DNK)': 'https://www.football-data.co.uk/new/DNK.csv',
    'Veikkausliiga (FIN)': 'https://www.football-data.co.uk/new/FIN.csv',
    'Premier Division (IRL)': 'https://www.football-data.co.uk/new/IRL.csv',
    'J-League (JPN)': 'https://www.football-data.co.uk/new/JPN.csv',
    'Liga MX (MEX)': 'https://www.football-data.co.uk/new/MEX.csv',
    'Eliteserien (NOR)': 'https://www.football-data.co.uk/new/NOR.csv',
    'Ekstraklasa (POL)': 'https://www.football-data.co.uk/new/POL.csv',
    'Liga 1 (ROU)': 'https://www.football-data.co.uk/new/ROU.csv',
    'Premier League (RUS)': 'https://www.football-data.co.uk/new/RUS.csv',
    'Allsvenskan (SWE)': 'https://www.football-data.co.uk/new/SWE.csv',
    'Super League (SWZ)': 'https://www.football-data.co.uk/new/SWZ.csv',
    'MLS (USA)': 'https://www.football-data.co.uk/new/USA.csv',
}

os.makedirs('cache', exist_ok=True)

# ============================================================================
# FUNCIONES BACKEND (TUS MISMAS FUNCIONES)
# ============================================================================

@st.cache_data(ttl=14400)  # Cache 4 horas
def cargar_datos_liga(liga):
    url = LIGAS_URLS.get(liga)
    if not url:
        return None
    
    try:
        df = pd.read_csv(url, encoding='latin-1')
        cols_necesarias = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
        cols_opcionales = ['HTHG', 'HTAG', 'HS', 'AS', 'HC', 'AC']
        cols = [c for c in cols_necesarias + cols_opcionales if c in df.columns]
        df = df[cols].dropna(subset=['FTHG', 'FTAG'])
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Date'])
        df['FTHG'] = pd.to_numeric(df['FTHG'], errors='coerce')
        df['FTAG'] = pd.to_numeric(df['FTAG'], errors='coerce')
        df = df.dropna(subset=['FTHG', 'FTAG'])
        
        if 'HC' in df.columns:
            df['HC'] = pd.to_numeric(df['HC'], errors='coerce').fillna(0)
        if 'AC' in df.columns:
            df['AC'] = pd.to_numeric(df['AC'], errors='coerce').fillna(0)
        
        modelo = calcular_dixon_coles_rapido(df)
        
        if modelo:
            return {
                'df': df,
                'modelo': modelo,
                'equipos': sorted(list(set(df['HomeTeam']) | set(df['AwayTeam']))),
                'fecha_actualizacion': datetime.now()
            }
        return None
    except Exception as e:
        st.error(f"Error cargando {liga}: {e}")
        return None

def calcular_dixon_coles_rapido(df):
    equipos = sorted(set(df['HomeTeam']) | set(df['AwayTeam']))
    n = len(equipos)
    
    if n < 4 or len(df) < MIN_PARTIDOS:
        return None
    
    eq_idx = {e: i for i, e in enumerate(equipos)}
    df_reciente = df.tail(50)
    params_iniciales = np.concatenate([[0.3], np.zeros(n), np.zeros(n)])
    
    def log_likelihood_simple(params):
        ha, ataque, defensa = params[0], params[1:n+1], params[n+1:]
        ll = 0
        for _, row in df_reciente.iterrows():
            try:
                home_idx, away_idx = eq_idx[row['HomeTeam']], eq_idx[row['AwayTeam']]
                lambda_h = np.exp(ha + ataque[home_idx] + defensa[away_idx])
                lambda_a = np.exp(ataque[away_idx] + defensa[home_idx])
                prob = poisson.pmf(int(row['FTHG']), lambda_h) * poisson.pmf(int(row['FTAG']), lambda_a)
                ll += np.log(prob + 1e-10)
            except:
                continue
        return -ll
    
    try:
        resultado = minimize(log_likelihood_simple, params_iniciales, method='L-BFGS-B',
                           options={'maxiter': 50, 'disp': False})
        ha, ataque, defensa = resultado.x[0], resultado.x[1:n+1], resultado.x[n+1:]
        params_df = pd.DataFrame({'Equipo': equipos, 'Ataque': ataque, 'Defensa': defensa})
        
        stats = {}
        for equipo in equipos:
            p_local = df[df['HomeTeam'] == equipo]
            p_visitante = df[df['AwayTeam'] == equipo]
            total = len(p_local) + len(p_visitante)
            if total > 0:
                stats[equipo] = {
                    'goles_favor': (p_local['FTHG'].sum() + p_visitante['FTAG'].sum()) / total,
                    'goles_contra': (p_local['FTAG'].sum() + p_visitante['FTHG'].sum()) / total,
                    'partidos': total,
                    'corners_prom': ((p_local['HC'].sum() if 'HC' in p_local.columns else 0) + 
                                    (p_visitante['AC'].sum() if 'AC' in p_visitante.columns else 0)) / max(total, 1)
                }
        return {'params': params_df, 'ha': ha, 'stats': stats}
    except:
        return None

def analizar_partido_completo(liga, equipo_local, equipo_visitante):
    """TU MISMA FUNCIÓN - Copio exactamente tu lógica"""
    datos = cargar_datos_liga(liga)
    if not datos or not datos['modelo']:
        return None
    
    df, modelo = datos['df'], datos['modelo']
    params, ha, stats = modelo['params'], modelo['ha'], modelo['stats']
    
    if equipo_local not in params['Equipo'].values or equipo_visitante not in params['Equipo'].values:
        return None
    
    try:
        ath = params[params['Equipo'] == equipo_local]['Ataque'].values[0]
        deh = params[params['Equipo'] == equipo_local]['Defensa'].values[0]
        ata = params[params['Equipo'] == equipo_visitante]['Ataque'].values[0]
        dea = params[params['Equipo'] == equipo_visitante]['Defensa'].values[0]
    except:
        return None
    
    lambda_h, lambda_a = np.exp(ha + ath + dea), np.exp(ata + deh)
    
    max_goles = 10
    matriz = np.array([[poisson.pmf(i, lambda_h) * poisson.pmf(j, lambda_a) 
                       for j in range(max_goles)] for i in range(max_goles)])
    
    lambda_h_ht, lambda_a_ht = lambda_h * 0.45, lambda_a * 0.45
    matriz_ht = np.array([[poisson.pmf(i, lambda_h_ht) * poisson.pmf(j, lambda_a_ht) 
                          for j in range(8)] for i in range(8)])
    
    # Todas tus probabilidades exactamente iguales
    prob_over_05_ht = (1 - matriz_ht[0, 0]) * 100
    prob_over_15_ht = sum(matriz_ht[i, j] for i in range(8) for j in range(8) if i + j > 1.5) * 100
    prob_btts_si = (1 - (poisson.pmf(0, lambda_h) + poisson.pmf(0, lambda_a) - 
                         poisson.pmf(0, lambda_h) * poisson.pmf(0, lambda_a))) * 100
    prob_over_15 = sum(matriz[i, j] for i in range(max_goles) for j in range(max_goles) if i + j > 1.5) * 100
    prob_over_25 = sum(matriz[i, j] for i in range(max_goles) for j in range(max_goles) if i + j > 2.5) * 100
    prob_over_35 = sum(matriz[i, j] for i in range(max_goles) for j in range(max_goles) if i + j > 3.5) * 100
    prob_local = np.sum(np.tril(matriz, -1)) * 100
    prob_empate = np.sum(np.diag(matriz)) * 100
    prob_visitante = np.sum(np.triu(matriz, 1)) * 100
    prob_local_minus_15 = sum(matriz[i, j] for i in range(max_goles) for j in range(max_goles) if i - j >= 2) * 100
    
    corners_local = stats.get(equipo_local, {}).get('corners_prom', 5)
    corners_visitante = stats.get(equipo_visitante, {}).get('corners_prom', 5)
    corners_esperados = corners_local + corners_visitante
    
    # HT/FT
    prob_local_ht = np.sum(np.tril(matriz_ht, -1)) * 100
    prob_empate_ht = np.sum(np.diag(matriz_ht)) * 100
    prob_visitante_ht = np.sum(np.triu(matriz_ht, 1)) * 100
    
    ht_ft = {
        '1/1': (prob_local_ht * prob_local) / 10000 * 100,
        '1/X': (prob_local_ht * prob_empate) / 10000 * 100,
        '1/2': (prob_local_ht * prob_visitante) / 10000 * 100,
        'X/1': (prob_empate_ht * prob_local) / 10000 * 100,
        'X/X': (prob_empate_ht * prob_empate) / 10000 * 100,
        'X/2': (prob_empate_ht * prob_visitante) / 10000 * 100,
        '2/1': (prob_visitante_ht * prob_local) / 10000 * 100,
        '2/X': (prob_visitante_ht * prob_empate) / 10000 * 100,
        '2/2': (prob_visitante_ht * prob_visitante) / 10000 * 100
    }
    
    return {
        'goles_esperados_total': lambda_h + lambda_a,
        'prob_over_05_ht': prob_over_05_ht,
        'prob_over_15_ht': prob_over_15_ht,
        'prob_btts_si': prob_btts_si,
        'prob_over_15': prob_over_15,
        'prob_over_25': prob_over_25,
        'prob_over_35': prob_over_35,
        'prob_local': prob_local,
        'prob_empate': prob_empate,
        'prob_visitante': prob_visitante,
        'cuota_local': 100 / prob_local if prob_local > 0 else 999,
        'cuota_empate': 100 / prob_empate if prob_empate > 0 else 999,
        'cuota_visitante': 100 / prob_visitante if prob_visitante > 0 else 999,
        'prob_local_minus_15': prob_local_minus_15,
        'prob_1x': prob_local + prob_empate,
        'prob_x2': prob_empate + prob_visitante,
        'prob_12': prob_local + prob_visitante,
        'corners_esperados': corners_esperados,
        'prob_over_85_corners': (1 - poisson.cdf(8, corners_esperados)) * 100,
        'ht_ft': ht_ft
    }

# ============================================================================
# INTERFAZ WEB
# ============================================================================

# CSS personalizado
st.markdown("""
<style>
    .main {background-color: #0e1117;}
    .stButton>button {
        background-color: #c9302c;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .big-font {
        font-size: 32px !important;
        font-weight: bold;
        color: #c9302c;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="big-font">🏆 D-GOL ULTIMATE 2025 🏆</p>', unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Análisis Profesional con Dixon-Coles + ML</h3>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuración")
    
    liga_seleccionada = st.selectbox(
        "🌍 Liga:",
        list(LIGAS_URLS.keys()),
        index=0
    )
    
    with st.spinner("Cargando equipos..."):
        datos = cargar_datos_liga(liga_seleccionada)
    
    if datos:
        st.success(f"✅ {len(datos['equipos'])} equipos")
        
        equipo_local = st.selectbox("🏠 Local:", datos['equipos'])
        equipo_visitante = st.selectbox("✈️ Visitante:", [e for e in datos['equipos'] if e != equipo_local])
        
        analizar_btn = st.button("🔍 ANALIZAR", type="primary", use_container_width=True)
    else:
        st.error("❌ Error cargando liga")
        analizar_btn = False

# Main
if analizar_btn and datos:
    with st.spinner("🔍 Analizando..."):
        resultado = analizar_partido_completo(liga_seleccionada, equipo_local, equipo_visitante)
    
    if resultado:
        # Header
        st.markdown(f"## 🏟️ {equipo_local} vs {equipo_visitante}")
        st.markdown(f"**Liga:** {liga_seleccionada}")
        st.markdown("---")
        
        # Goles esperados
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.metric("⚽ GOLES ESPERADOS", f"{resultado['goles_esperados_total']:.2f}")
        
        st.markdown("---")
        
        # 1X2
        st.subheader("📊 Probabilidades 1X2")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🏠 Local", f"{resultado['prob_local']:.1f}%")
            st.info(f"Cuota: {resultado['cuota_local']:.2f}")
        with col2:
            st.metric("⚖️ Empate", f"{resultado['prob_empate']:.1f}%")
            st.info(f"Cuota: {resultado['cuota_empate']:.2f}")
        with col3:
            st.metric("✈️ Visitante", f"{resultado['prob_visitante']:.1f}%")
            st.info(f"Cuota: {resultado['cuota_visitante']:.2f}")
        
        st.markdown("---")
        
        # Over/Under
        st.subheader("📈 Over/Under")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Over 1.5", f"{resultado['prob_over_15']:.1f}%")
        with col2:
            st.metric("Over 2.5", f"{resultado['prob_over_25']:.1f}%")
        with col3:
            st.metric("Over 3.5", f"{resultado['prob_over_35']:.1f}%")
        
        # BTTS
        st.markdown("---")
        st.subheader("⚽⚽ BTTS")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sí", f"{resultado['prob_btts_si']:.1f}%")
        with col2:
            st.metric("No", f"{100 - resultado['prob_btts_si']:.1f}%")
        
        # Corners
        st.markdown("---")
        st.subheader("🚩 Corners")
        st.metric("Esperados", f"{resultado['corners_esperados']:.1f}")
        st.metric("Over 8.5", f"{resultado['prob_over_85_corners']:.1f}%")
        
        # HT/FT
        st.markdown("---")
        st.subheader("📋 Descanso / Final")
        cols = st.columns(3)
        htft_keys = ['1/1', '1/X', '1/2', 'X/1', 'X/X', 'X/2', '2/1', '2/X', '2/2']
        for idx, key in enumerate(htft_keys):
            with cols[idx % 3]:
                st.metric(key, f"{resultado['ht_ft'][key]:.1f}%")
        
        # Recomendaciones
        st.markdown("---")
        st.subheader("💡 Recomendaciones")
        if resultado['prob_over_25'] > 70:
            st.success("✅ Over 2.5 - ALTA probabilidad")
        if resultado['prob_btts_si'] > 70:
            st.success("✅ BTTS Sí - ALTA probabilidad")
        if resultado['prob_local'] > 60:
            st.success(f"✅ Victoria {equipo_local} - Favorito")
        
        st.caption(f"⏰ {datetime.now().strftime('%d/%m/%Y %H:%M')} | Dixon-Coles + ML")
    else:
        st.error("❌ Error en el análisis")
else:
    st.info("👈 Selecciona liga y equipos para comenzar")
