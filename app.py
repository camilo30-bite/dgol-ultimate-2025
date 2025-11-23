"""
🏆 D-GOL ULTIMATE 2025 v3.0 - VERSIÓN DEFINITIVA 🏆
✅ Factor LOCAL/VISITANTE integrado
✅ Tarjetas por equipo (datos reales)
✅ Corners múltiples líneas
✅ Goles por equipo
✅ Análisis completo de rendimiento casa/fuera
✅ Fuente Inter mejorada
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize
import requests
import os
from datetime import datetime
import warnings
import plotly.graph_objects as go
import plotly.express as px

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

MIN_PARTIDOS = 15

# ===== 38 LIGAS COMPLETAS =====
LIGAS_URLS = {
    # Inglaterra (5)
    'Premier League (ENG)': 'https://www.football-data.co.uk/mmz4281/2526/E0.csv',
    'Championship (ENG)': 'https://www.football-data.co.uk/mmz4281/2526/E1.csv',
    'League One (ENG)': 'https://www.football-data.co.uk/mmz4281/2526/E2.csv',
    'League Two (ENG)': 'https://www.football-data.co.uk/mmz4281/2526/E3.csv',
    'National League (ENG)': 'https://www.football-data.co.uk/mmz4281/2526/EC.csv',
    # Escocia (4)
    'Premiership (SCO)': 'https://www.football-data.co.uk/mmz4281/2526/SC0.csv',
    'Championship (SCO)': 'https://www.football-data.co.uk/mmz4281/2526/SC1.csv',
    'League One (SCO)': 'https://www.football-data.co.uk/mmz4281/2526/SC2.csv',
    'League Two (SCO)': 'https://www.football-data.co.uk/mmz4281/2526/SC3.csv',
    # España (2)
    'La Liga (ESP)': 'https://www.football-data.co.uk/mmz4281/2526/SP1.csv',
    'Segunda División (ESP)': 'https://www.football-data.co.uk/mmz4281/2526/SP2.csv',
    # Italia (2)
    'Serie A (ITA)': 'https://www.football-data.co.uk/mmz4281/2526/I1.csv',
    'Serie B (ITA)': 'https://www.football-data.co.uk/mmz4281/2526/I2.csv',
    # Alemania (2)
    'Bundesliga (GER)': 'https://www.football-data.co.uk/mmz4281/2526/D1.csv',
    'Bundesliga 2 (GER)': 'https://www.football-data.co.uk/mmz4281/2526/D2.csv',
    # Francia (2)
    'Ligue 1 (FRA)': 'https://www.football-data.co.uk/mmz4281/2526/F1.csv',
    'Ligue 2 (FRA)': 'https://www.football-data.co.uk/mmz4281/2526/F2.csv',
    # Holanda (1)
    'Eredivisie (NED)': 'https://www.football-data.co.uk/mmz4281/2526/N1.csv',
    # Portugal (1)
    'Primeira Liga (POR)': 'https://www.football-data.co.uk/mmz4281/2526/P1.csv',
    # Bélgica (1)
    'Jupiler Pro League (BEL)': 'https://www.football-data.co.uk/mmz4281/2526/B1.csv',
    # Turquía (1)
    'Süper Lig (TUR)': 'https://www.football-data.co.uk/mmz4281/2526/T1.csv',
    # Grecia (1)
    'Super League (GRE)': 'https://www.football-data.co.uk/mmz4281/2526/G1.csv',
    # Resto del mundo (16)
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

# ============================================================================
# FUNCIONES BACKEND CON FACTOR LOCAL/VISITANTE
# ============================================================================

@st.cache_data(ttl=14400)
def cargar_datos_liga(liga):
    url = LIGAS_URLS.get(liga)
    if not url:
        return None
    
    try:
        df = pd.read_csv(url, encoding='latin-1')
        
        cols_necesarias = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
        cols_opcionales = ['HTHG', 'HTAG', 'HS', 'AS', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']
        cols = [c for c in cols_necesarias + cols_opcionales if c in df.columns]
        
        df = df[cols].dropna(subset=['FTHG', 'FTAG'])
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Date'])
        
        for col in ['FTHG', 'FTAG', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        modelo = calcular_modelo_mejorado(df)
        
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

def calcular_modelo_mejorado(df):
    """Modelo Dixon-Coles + estadísticas LOCAL/VISITANTE separadas"""
    equipos = sorted(set(df['HomeTeam']) | set(df['AwayTeam']))
    n = len(equipos)
    
    if n < 4 or len(df) < MIN_PARTIDOS:
        return None
    
    eq_idx = {e: i for i, e in enumerate(equipos)}
    df_reciente = df.tail(50)
    params_iniciales = np.concatenate([[0.3], np.zeros(n), np.zeros(n)])
    
    def log_likelihood(params):
        ha, ataque, defensa = params[0], params[1:n+1], params[n+1:]
        ll = 0
        for _, row in df_reciente.iterrows():
            try:
                h_idx, a_idx = eq_idx[row['HomeTeam']], eq_idx[row['AwayTeam']]
                lambda_h = np.exp(ha + ataque[h_idx] + defensa[a_idx])
                lambda_a = np.exp(ataque[a_idx] + defensa[h_idx])
                prob = poisson.pmf(int(row['FTHG']), lambda_h) * poisson.pmf(int(row['FTAG']), lambda_a)
                ll += np.log(prob + 1e-10)
            except:
                continue
        return -ll
    
    try:
        res = minimize(log_likelihood, params_iniciales, method='L-BFGS-B', options={'maxiter': 50})
        ha, ataque, defensa = res.x[0], res.x[1:n+1], res.x[n+1:]
        params_df = pd.DataFrame({'Equipo': equipos, 'Ataque': ataque, 'Defensa': defensa})
        
        # ESTADÍSTICAS SEPARADAS LOCAL/VISITANTE
        stats = {}
        for equipo in equipos:
            p_local = df[df['HomeTeam'] == equipo]
            p_visitante = df[df['AwayTeam'] == equipo]
            total_local = len(p_local)
            total_visit = len(p_visitante)
            total = total_local + total_visit
            
            if total > 0:
                # RENDIMIENTO LOCAL
                goles_local_casa = p_local['FTHG'].sum()
                goles_contra_local_casa = p_local['FTAG'].sum()
                victorias_local = len(p_local[p_local['FTR'] == 'H'])
                empates_local = len(p_local[p_local['FTR'] == 'D'])
                derrotas_local = len(p_local[p_local['FTR'] == 'A'])
                
                # RENDIMIENTO VISITANTE
                goles_visit_fuera = p_visitante['FTAG'].sum()
                goles_contra_visit_fuera = p_visitante['FTHG'].sum()
                victorias_visit = len(p_visitante[p_visitante['FTR'] == 'A'])
                empates_visit = len(p_visitante[p_visitante['FTR'] == 'D'])
                derrotas_visit = len(p_visitante[p_visitante['FTR'] == 'H'])
                
                # CORNERS LOCAL/VISITANTE
                if 'HC' in p_local.columns:
                    corners_local = p_local['HC'].sum() / total_local if total_local > 0 else 5
                    corners_visit = p_visitante['AC'].sum() / total_visit if total_visit > 0 else 5
                else:
                    corners_local = 5
                    corners_visit = 5
                
                # TARJETAS LOCAL/VISITANTE
                if 'HY' in p_local.columns:
                    tarj_local = (p_local['HY'].sum() + p_local.get('HR', pd.Series([0])).sum() * 2) / total_local if total_local > 0 else 2.5
                    tarj_visit = (p_visitante['AY'].sum() + p_visitante.get('AR', pd.Series([0])).sum() * 2) / total_visit if total_visit > 0 else 2.5
                else:
                    tarj_local = 2.5
                    tarj_visit = 2.5
                
                # CALCULAR ÍNDICES DE RENDIMIENTO
                ptos_local = (victorias_local * 3 + empates_local) / max(total_local * 3, 1) * 100
                ptos_visit = (victorias_visit * 3 + empates_visit) / max(total_visit * 3, 1) * 100
                
                stats[equipo] = {
                    # GLOBALES
                    'partidos_total': total,
                    'partidos_local': total_local,
                    'partidos_visit': total_visit,
                    
                    # LOCAL
                    'goles_favor_local': goles_local_casa / total_local if total_local > 0 else 0,
                    'goles_contra_local': goles_contra_local_casa / total_local if total_local > 0 else 0,
                    'victorias_local': victorias_local,
                    'empates_local': empates_local,
                    'derrotas_local': derrotas_local,
                    'ptos_local_pct': ptos_local,
                    'corners_local': corners_local,
                    'tarjetas_local': tarj_local,
                    
                    # VISITANTE
                    'goles_favor_visit': goles_visit_fuera / total_visit if total_visit > 0 else 0,
                    'goles_contra_visit': goles_contra_visit_fuera / total_visit if total_visit > 0 else 0,
                    'victorias_visit': victorias_visit,
                    'empates_visit': empates_visit,
                    'derrotas_visit': derrotas_visit,
                    'ptos_visit_pct': ptos_visit,
                    'corners_visit': corners_visit,
                    'tarjetas_visit': tarj_visit,
                    
                    # DIFERENCIALES
                    'diferencia_goles_local_visit': (goles_local_casa / max(total_local, 1)) - (goles_visit_fuera / max(total_visit, 1)),
                    'diferencia_ptos_local_visit': ptos_local - ptos_visit,
                }
        
        return {'params': params_df, 'ha': ha, 'stats': stats}
    except:
        return None

def analizar_partido_completo(liga, equipo_local, equipo_visitante):
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
    
    # OBTENER ESTADÍSTICAS LOCAL/VISITANTE
    stats_local = stats.get(equipo_local, {})
    stats_visit = stats.get(equipo_visitante, {})
    
    # AJUSTAR LAMBDA CON FACTOR LOCAL/VISITANTE
    lambda_h_base = np.exp(ha + ath + dea)
    lambda_a_base = np.exp(ata + deh)
    
    # Factor de ajuste basado en rendimiento local/visitante
    factor_local_goles = stats_local.get('goles_favor_local', 1.5) / max(stats_local.get('goles_favor_visit', 1.0), 0.5)
    factor_visit_goles = stats_visit.get('goles_favor_visit', 1.0) / max(stats_visit.get('goles_favor_local', 1.5), 0.5)
    
    # Aplicar ajuste moderado (15% máximo)
    ajuste_local = 1 + (factor_local_goles - 1) * 0.15
    ajuste_visit = 1 + (factor_visit_goles - 1) * 0.15
    
    lambda_h = lambda_h_base * ajuste_local
    lambda_a = lambda_a_base * ajuste_visit
    
    # Matrices de probabilidades
    max_goles = 10
    matriz = np.array([[poisson.pmf(i, lambda_h) * poisson.pmf(j, lambda_a) 
                       for j in range(max_goles)] for i in range(max_goles)])
    
    lambda_h_ht, lambda_a_ht = lambda_h * 0.45, lambda_a * 0.45
    matriz_ht = np.array([[poisson.pmf(i, lambda_h_ht) * poisson.pmf(j, lambda_a_ht) 
                          for j in range(8)] for i in range(8)])
    
    # Probabilidades básicas
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
    
    # GOLES POR EQUIPO
    prob_local_over_05 = (1 - poisson.pmf(0, lambda_h)) * 100
    prob_local_over_15 = (1 - poisson.cdf(1, lambda_h)) * 100
    prob_local_over_25 = (1 - poisson.cdf(2, lambda_h)) * 100
    prob_local_over_35 = (1 - poisson.cdf(3, lambda_h)) * 100
    
    prob_visit_over_05 = (1 - poisson.pmf(0, lambda_a)) * 100
    prob_visit_over_15 = (1 - poisson.cdf(1, lambda_a)) * 100
    prob_visit_over_25 = (1 - poisson.cdf(2, lambda_a)) * 100
    prob_visit_over_35 = (1 - poisson.cdf(3, lambda_a)) * 100
    
    # CORNERS usando estadísticas específicas local/visitante
    corners_local = stats_local.get('corners_local', 5)
    corners_visit = stats_visit.get('corners_visit', 5)
    corners_esperados = corners_local + corners_visit
    
    lineas_corners = {}
    for linea in [6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5]:
        lineas_corners[linea] = (1 - poisson.cdf(int(linea), corners_esperados)) * 100
    
    # TARJETAS usando estadísticas específicas local/visitante
    tarj_local = stats_local.get('tarjetas_local', 2.5)
    tarj_visit = stats_visit.get('tarjetas_visit', 2.5)
    tarjetas_esperadas = tarj_local + tarj_visit
    
    lineas_tarjetas = {}
    for linea in [2.5, 3.5, 4.5, 5.5, 6.5]:
        lineas_tarjetas[linea] = (1 - poisson.cdf(int(linea), tarjetas_esperadas)) * 100
    
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
        'lambda_local': lambda_h,
        'lambda_visitante': lambda_a,
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
        'local_over_05': prob_local_over_05,
        'local_over_15': prob_local_over_15,
        'local_over_25': prob_local_over_25,
        'local_over_35': prob_local_over_35,
        'visit_over_05': prob_visit_over_05,
        'visit_over_15': prob_visit_over_15,
        'visit_over_25': prob_visit_over_25,
        'visit_over_35': prob_visit_over_35,
        'corners_esperados': corners_esperados,
        'lineas_corners': lineas_corners,
        'tarjetas_esperadas': tarjetas_esperadas,
        'lineas_tarjetas': lineas_tarjetas,
        'tarj_local': tarj_local,
        'tarj_visitante': tarj_visit,
        'ht_ft': ht_ft,
        'matriz': matriz,
        'prob_1x': prob_local + prob_empate,
        'prob_x2': prob_empate + prob_visitante,
        'prob_12': prob_local + prob_visitante,
        'stats_local': stats_local,
        'stats_visit': stats_visit,
        'ajuste_aplicado': {
            'factor_local': ajuste_local,
            'factor_visit': ajuste_visit
        }
    }

# ============================================================================
# INTERFAZ MEJORADA
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .main {background-color: #0e1117;}
    
    .stButton>button {
        background-color: #c9302c;
        color: white;
        font-weight: 600;
        border-radius: 10px;
        padding: 12px 24px;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s;
        border: none;
    }
    
    .stButton>button:hover {
        background-color: #a12620;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(201, 48, 44, 0.4);
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
    }
    
    .stMetric {
        font-family: 'Inter', sans-serif;
    }
    
    .stMetric label {
        font-size: 15px !important;
        font-weight: 600 !important;
    }
    
    .stMetric .metric-value {
        font-size: 28px !important;
        font-weight: 700 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 style="text-align: center; color: #c9302c; font-size: 42px;">🏆 D-GOL ULTIMATE 2025 v3.0 🏆</h1>', unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; font-weight: 500;'>Análisis con Factor Local/Visitante + Dixon-Coles + ML</h3>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuración")
    
    if st.button("🔄 Actualizar Datos", use_container_width=True):
        st.cache_data.clear()
        st.success("✅ Caché limpiado")
        st.rerun()
    
    liga = st.selectbox("🌍 Liga:", list(LIGAS_URLS.keys()))
    
    with st.spinner("Cargando equipos..."):
        datos = cargar_datos_liga(liga)
    
    if datos:
        st.success(f"✅ {len(datos['equipos'])} equipos")
        st.info(f"📅 Actualizado: {datos['fecha_actualizacion'].strftime('%d/%m/%Y %H:%M')}")
        st.caption("♻️ Se actualiza cada 4 horas")
        
        local = st.selectbox("🏠 Local:", datos['equipos'])
        visitante = st.selectbox("✈️ Visitante:", [e for e in datos['equipos'] if e != local])
        analizar_btn = st.button("🔍 ANALIZAR PARTIDO", type="primary", use_container_width=True)
    else:
        st.error("❌ Error cargando liga")
        analizar_btn = False

# Main
if analizar_btn and datos:
    with st.spinner("🔍 Analizando partido con factor local/visitante..."):
        resultado = analizar_partido_completo(liga, local, visitante)
    
    if resultado:
        # Header partido
        st.markdown(f"## 🏟️ {local} vs {visitante}")
        st.markdown(f"**Liga:** {liga}")
        
        # Ajuste aplicado
        ajuste = resultado.get('ajuste_aplicado', {})
        if ajuste.get('factor_local', 1) != 1 or ajuste.get('factor_visit', 1) != 1:
            st.caption(f"⚙️ Ajuste local/visitante aplicado: Local x{ajuste.get('factor_local', 1):.2f}, Visitante x{ajuste.get('factor_visit', 1):.2f}")
        
        st.markdown("---")
        
        # Goles esperados
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.metric("⚽ GOLES ESPERADOS TOTAL", f"{resultado['goles_esperados_total']:.2f}",
                     delta=f"🏠 {resultado['lambda_local']:.2f} | ✈️ {resultado['lambda_visitante']:.2f}")
        
        st.markdown("---")
        
        # TABS
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 Probabilidades", 
            "⚽ Goles por Equipo", 
            "🚩 Corners", 
            "🟨 Tarjetas",
            "🏠 Local vs ✈️ Visitante",
            "📈 Gráficos", 
            "💾 Exportar"
        ])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Resultados Descanso")
                st.metric("Over 0.5 HT", f"{resultado['prob_over_05_ht']:.1f}%")
                st.metric("Over 1.5 HT", f"{resultado['prob_over_15_ht']:.1f}%")
                
                st.markdown("---")
                st.subheader("⚽⚽ BTTS (Ambos Marcan)")
                st.metric("Sí", f"{resultado['prob_btts_si']:.1f}%")
                st.metric("No", f"{100 - resultado['prob_btts_si']:.1f}%")
                
                st.markdown("---")
                st.subheader("📈 Over/Under Total")
                st.metric("Over 1.5", f"{resultado['prob_over_15']:.1f}%")
                st.metric("Over 2.5", f"{resultado['prob_over_25']:.1f}%")
                st.metric("Over 3.5", f"{resultado['prob_over_35']:.1f}%")
            
            with col2:
                st.subheader("📊 Resultado Final 1X2")
                col_1, col_x, col_2 = st.columns(3)
                with col_1:
                    st.metric("🏠 Local", f"{resultado['prob_local']:.1f}%")
                    st.info(f"Cuota: {resultado['cuota_local']:.2f}")
                with col_x:
                    st.metric("⚖️ Empate", f"{resultado['prob_empate']:.1f}%")
                    st.info(f"Cuota: {resultado['cuota_empate']:.2f}")
                with col_2:
                    st.metric("✈️ Visitante", f"{resultado['prob_visitante']:.1f}%")
                    st.info(f"Cuota: {resultado['cuota_visitante']:.2f}")
                
                st.markdown("---")
                st.subheader("🤝 Doble Oportunidad")
                st.metric("1X (Local o Empate)", f"{resultado['prob_1x']:.1f}%")
                st.metric("X2 (Empate o Visitante)", f"{resultado['prob_x2']:.1f}%")
                st.metric("12 (Local o Visitante)", f"{resultado['prob_12']:.1f}%")
        
        with tab2:
            st.subheader("⚽ GOLES POR EQUIPO")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### 🏠 {local}")
                st.metric(f"Goles esperados", f"{resultado['lambda_local']:.2f}")
                st.markdown("---")
                st.metric("Over 0.5 goles", f"{resultado['local_over_05']:.1f}%")
                st.metric("Over 1.5 goles", f"{resultado['local_over_15']:.1f}%")
                st.metric("Over 2.5 goles", f"{resultado['local_over_25']:.1f}%")
                st.metric("Over 3.5 goles", f"{resultado['local_over_35']:.1f}%")
            
            with col2:
                st.markdown(f"### ✈️ {visitante}")
                st.metric(f"Goles esperados", f"{resultado['lambda_visitante']:.2f}")
                st.markdown("---")
                st.metric("Over 0.5 goles", f"{resultado['visit_over_05']:.1f}%")
                st.metric("Over 1.5 goles", f"{resultado['visit_over_15']:.1f}%")
                st.metric("Over 2.5 goles", f"{resultado['visit_over_25']:.1f}%")
                st.metric("Over 3.5 goles", f"{resultado['visit_over_35']:.1f}%")
            
            st.markdown("---")
            st.info("💡 **Tip:** Estos mercados analizan los goles de CADA equipo por separado, útil para apuestas específicas por equipo.")
        
        with tab3:
            st.subheader("🚩 ANÁLISIS DE CORNERS")
            
            st.metric("📊 Corners Esperados", f"{resultado['corners_esperados']:.1f}")
            
            st.markdown("---")
            st.markdown("### 📈 Probabilidades por Línea")
            
            col1, col2, col3 = st.columns(3)
            lineas = list(resultado['lineas_corners'].items())
            
            for idx, (linea, prob) in enumerate(lineas):
                with [col1, col2, col3][idx % 3]:
                    st.metric(f"Over {linea}", f"{prob:.1f}%")
            
            st.markdown("---")
            
            fig_corners = go.Figure()
            fig_corners.add_trace(go.Bar(
                x=[f"Over {l}" for l in resultado['lineas_corners'].keys()],
                y=list(resultado['lineas_corners'].values()),
                marker_color='#5bc0de',
                text=[f"{v:.1f}%" for v in resultado['lineas_corners'].values()],
                textposition='outside'
            ))
            fig_corners.update_layout(
                title='Probabilidades de Corners',
                yaxis_title='Probabilidad (%)',
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_corners, use_container_width=True)
        
        with tab4:
            st.subheader("🟨 ANÁLISIS DE TARJETAS")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Tarjetas Esperadas", f"{resultado['tarjetas_esperadas']:.1f}")
            with col2:
                st.metric(f"🏠 {local}", f"{resultado['tarj_local']:.1f}")
            with col3:
                st.metric(f"✈️ {visitante}", f"{resultado['tarj_visitante']:.1f}")
            
            st.markdown("---")
            st.markdown("### 📈 Probabilidades por Línea")
            
            col1, col2, col3 = st.columns(3)
            lineas_tarj = list(resultado['lineas_tarjetas'].items())
            
            for idx, (linea, prob) in enumerate(lineas_tarj):
                with [col1, col2, col3][idx % 3]:
                    st.metric(f"Over {linea}", f"{prob:.1f}%")
            
            st.markdown("---")
            st.info("💡 **Nota:** Tarjetas = Amarillas + (Rojas × 2). Datos reales de cada equipo jugando local/visitante.")
            
            fig_tarj = go.Figure()
            fig_tarj.add_trace(go.Bar(
                x=[f"Over {l}" for l in resultado['lineas_tarjetas'].keys()],
                y=list(resultado['lineas_tarjetas'].values()),
                marker_color='#f0ad4e',
                text=[f"{v:.1f}%" for v in resultado['lineas_tarjetas'].values()],
                textposition='outside'
            ))
            fig_tarj.update_layout(
                title='Probabilidades de Tarjetas',
                yaxis_title='Probabilidad (%)',
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_tarj, use_container_width=True)
        
        with tab5:
            st.subheader("🏠 Análisis Local vs ✈️ Visitante")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### 🏠 {local} (LOCAL)")
                
                stats_l = resultado['stats_local']
                
                st.metric("Partidos en casa", stats_l.get('partidos_local', 0))
                st.metric("Goles a favor (casa)", f"{stats_l.get('goles_favor_local', 0):.2f} por partido")
                st.metric("Goles en contra (casa)", f"{stats_l.get('goles_contra_local', 0):.2f} por partido")
                
                st.markdown("---")
                st.markdown("**Resultados en casa:**")
                vic_l = stats_l.get('victorias_local', 0)
                emp_l = stats_l.get('empates_local', 0)
                der_l = stats_l.get('derrotas_local', 0)
                st.metric("Victorias", vic_l)
                st.metric("Empates", emp_l)
                st.metric("Derrotas", der_l)
                st.metric("% Puntos obtenidos", f"{stats_l.get('ptos_local_pct', 0):.1f}%")
                
                st.markdown("---")
                st.metric("Corners promedio (casa)", f"{stats_l.get('corners_local', 0):.1f}")
                st.metric("Tarjetas promedio (casa)", f"{stats_l.get('tarjetas_local', 0):.1f}")
            
            with col2:
                st.markdown(f"### ✈️ {visitante} (VISITANTE)")
                
                stats_v = resultado['stats_visit']
                
                st.metric("Partidos fuera", stats_v.get('partidos_visit', 0))
                st.metric("Goles a favor (fuera)", f"{stats_v.get('goles_favor_visit', 0):.2f} por partido")
                st.metric("Goles en contra (fuera)", f"{stats_v.get('goles_contra_visit', 0):.2f} por partido")
                
                st.markdown("---")
                st.markdown("**Resultados fuera:**")
                vic_v = stats_v.get('victorias_visit', 0)
                emp_v = stats_v.get('empates_visit', 0)
                der_v = stats_v.get('derrotas_visit', 0)
                st.metric("Victorias", vic_v)
                st.metric("Empates", emp_v)
                st.metric("Derrotas", der_v)
                st.metric("% Puntos obtenidos", f"{stats_v.get('ptos_visit_pct', 0):.1f}%")
                
                st.markdown("---")
                st.metric("Corners promedio (fuera)", f"{stats_v.get('corners_visit', 0):.1f}")
                st.metric("Tarjetas promedio (fuera)", f"{stats_v.get('tarjetas_visit', 0):.1f}")
            
            st.markdown("---")
            
            # Comparación visual
            st.subheader("📊 Comparación Visual")
            
            fig_comp = go.Figure()
            
            categorias = ['Goles Favor', 'Goles Contra', '% Puntos', 'Corners', 'Tarjetas']
            valores_local = [
                stats_l.get('goles_favor_local', 0),
                stats_l.get('goles_contra_local', 0),
                stats_l.get('ptos_local_pct', 0) / 10,
                stats_l.get('corners_local', 0),
                stats_l.get('tarjetas_local', 0)
            ]
            valores_visit = [
                stats_v.get('goles_favor_visit', 0),
                stats_v.get('goles_contra_visit', 0),
                stats_v.get('ptos_visit_pct', 0) / 10,
                stats_v.get('corners_visit', 0),
                stats_v.get('tarjetas_visit', 0)
            ]
            
            fig_comp.add_trace(go.Bar(name=f'{local} (Casa)', x=categorias, y=valores_local, marker_color='#c9302c'))
            fig_comp.add_trace(go.Bar(name=f'{visitante} (Fuera)', x=categorias, y=valores_visit, marker_color='#5bc0de'))
            
            fig_comp.update_layout(title='Rendimiento Local vs Visitante', barmode='group', height=400)
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # Insights
            st.markdown("---")
            st.subheader("💡 Insights")
            
            dif_goles = stats_l.get('diferencia_goles_local_visit', 0)
            
            if dif_goles > 0.5:
                st.success(f"✅ {local} es significativamente mejor en casa (+{dif_goles:.2f} goles/partido)")
            elif dif_goles < -0.5:
                st.warning(f"⚠️ {local} rinde menos en casa ({dif_goles:.2f} goles/partido)")
            else:
                st.info(f"ℹ️ {local} tiene rendimiento similar local/visitante")
            
            ptos_visit = stats_v.get('ptos_visit_pct', 0)
            if ptos_visit < 30:
                st.error(f"❌ {visitante} tiene mal rendimiento fuera (solo {ptos_visit:.1f}% puntos)")
            elif ptos_visit > 50:
                st.success(f"✅ {visitante} es sólido fuera ({ptos_visit:.1f}% puntos)")
            else:
                st.info(f"ℹ️ {visitante} tiene rendimiento promedio fuera ({ptos_visit:.1f}% puntos)")
        
        with tab6:
            st.subheader("📊 Visualizaciones")
            
            # Gráfico 1X2
            fig1 = go.Figure(data=[
                go.Bar(name='Probabilidad', x=['Local', 'Empate', 'Visitante'],
                      y=[resultado['prob_local'], resultado['prob_empate'], resultado['prob_visitante']],
                      marker_color=['#c9302c', '#f0ad4e', '#5bc0de'],
                      text=[f"{resultado['prob_local']:.1f}%", f"{resultado['prob_empate']:.1f}%", f"{resultado['prob_visitante']:.1f}%"],
                      textposition='outside')
            ])
            fig1.update_layout(title='Probabilidades 1X2', yaxis_title='Probabilidad (%)', height=400, showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)
            
            # Matriz resultados exactos
            st.subheader("🎯 Matriz de Resultados Exactos (Top 6x6)")
            matriz_display = resultado['matriz'][:6, :6] * 100
            fig3 = px.imshow(matriz_display,
                            labels=dict(x="Goles Visitante", y="Goles Local", color="Prob. (%)"),
                            x=[str(i) for i in range(6)],
                            y=[str(i) for i in range(6)],
                            color_continuous_scale='Reds',
                            text_auto='.1f')
            fig3.update_layout(height=500)
            st.plotly_chart(fig3, use_container_width=True)
        
        with tab7:
            st.subheader("💾 Exportar Análisis")
            
            reporte = f"""
D-GOL ULTIMATE 2025 v3.0 - ANÁLISIS COMPLETO CON FACTOR LOCAL/VISITANTE
{"="*75}

Partido: {local} vs {visitante}
Liga: {liga}
Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}

{"="*75}

AJUSTE LOCAL/VISITANTE APLICADO:
- Factor Local: x{resultado['ajuste_aplicado']['factor_local']:.2f}
- Factor Visitante: x{resultado['ajuste_aplicado']['factor_visit']:.2f}

GOLES ESPERADOS:
- Total: {resultado['goles_esperados_total']:.2f}
- Local: {resultado['lambda_local']:.2f}
- Visitante: {resultado['lambda_visitante']:.2f}

1X2:
- Local: {resultado['prob_local']:.1f}% (Cuota: {resultado['cuota_local']:.2f})
- Empate: {resultado['prob_empate']:.1f}% (Cuota: {resultado['cuota_empate']:.2f})
- Visitante: {resultado['prob_visitante']:.1f}% (Cuota: {resultado['cuota_visitante']:.2f})

GOLES POR EQUIPO - {local}:
- Over 0.5: {resultado['local_over_05']:.1f}%
- Over 1.5: {resultado['local_over_15']:.1f}%
- Over 2.5: {resultado['local_over_25']:.1f}%
- Over 3.5: {resultado['local_over_35']:.1f}%

GOLES POR EQUIPO - {visitante}:
- Over 0.5: {resultado['visit_over_05']:.1f}%
- Over 1.5: {resultado['visit_over_15']:.1f}%
- Over 2.5: {resultado['visit_over_25']:.1f}%
- Over 3.5: {resultado['visit_over_35']:.1f}%

CORNERS:
- Esperados: {resultado['corners_esperados']:.1f}
"""
            for linea, prob in resultado['lineas_corners'].items():
                reporte += f"- Over {linea}: {prob:.1f}%\n"
            
            reporte += f"""
TARJETAS:
- Esperadas: {resultado['tarjetas_esperadas']:.1f}
- {local} (casa): {resultado['tarj_local']:.1f}
- {visitante} (fuera): {resultado['tarj_visitante']:.1f}
"""
            for linea, prob in resultado['lineas_tarjetas'].items():
                reporte += f"- Over {linea}: {prob:.1f}%\n"
            
            stats_l = resultado['stats_local']
            stats_v = resultado['stats_visit']
            
            reporte += f"""
RENDIMIENTO LOCAL - {local}:
- Partidos: {stats_l.get('partidos_local', 0)}
- Goles favor: {stats_l.get('goles_favor_local', 0):.2f}/partido
- Goles contra: {stats_l.get('goles_contra_local', 0):.2f}/partido
- Victorias: {stats_l.get('victorias_local', 0)}
- Empates: {stats_l.get('empates_local', 0)}
- Derrotas: {stats_l.get('derrotas_local', 0)}
- % Puntos: {stats_l.get('ptos_local_pct', 0):.1f}%

RENDIMIENTO VISITANTE - {visitante}:
- Partidos: {stats_v.get('partidos_visit', 0)}
- Goles favor: {stats_v.get('goles_favor_visit', 0):.2f}/partido
- Goles contra: {stats_v.get('goles_contra_visit', 0):.2f}/partido
- Victorias: {stats_v.get('victorias_visit', 0)}
- Empates: {stats_v.get('empates_visit', 0)}
- Derrotas: {stats_v.get('derrotas_visit', 0)}
- % Puntos: {stats_v.get('ptos_visit_pct', 0):.1f}%

BTTS:
- Sí: {resultado['prob_btts_si']:.1f}%
- No: {100 - resultado['prob_btts_si']:.1f}%

{"="*75}
Generado por D-GOL Ultimate 2025 v3.0
Modelo: Dixon-Coles + Factor Local/Visitante + Machine Learning
            """
            
            st.download_button(
                label="💾 Descargar Análisis Completo (TXT)",
                data=reporte,
                file_name=f"dgol_v3_{local}_vs_{visitante}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        # Recomendaciones inteligentes
        st.markdown("---")
        st.subheader("💡 Recomendaciones Inteligentes (Factor L/V incluido)")
        
        recomendaciones = []
        
        if resultado['prob_over_25'] > 70:
            recomendaciones.append(("✅ Over 2.5 Total", "Alta probabilidad", "success"))
        
        if resultado['prob_btts_si'] > 70:
            recomendaciones.append(("✅ BTTS Sí", "Muy probable", "success"))
        
        if resultado['local_over_15'] > 70:
            recomendaciones.append((f"✅ {local} Over 1.5", "Alta probabilidad", "success"))
        
        if resultado['visit_over_15'] > 70:
            recomendaciones.append((f"✅ {visitante} Over 1.5", "Alta probabilidad", "success"))
        
        if resultado['lineas_corners'].get(8.5, 0) > 70:
            recomendaciones.append(("✅ Over 8.5 Corners", "Alta probabilidad", "success"))
        
        # Recomendaciones basadas en factor local/visitante
        stats_l = resultado['stats_local']
        stats_v = resultado['stats_visit']
        
        if stats_l.get('ptos_local_pct', 0) > 60 and stats_v.get('ptos_visit_pct', 0) < 35:
            recomendaciones.append((f"✅ Victoria {local}", "Fuerte en casa vs débil fuera", "success"))
        
        if stats_l.get('goles_favor_local', 0) > 2.0 and stats_v.get('goles_contra_visit', 0) > 1.5:
            recomendaciones.append((f"✅ {local} Over 1.5 goles", "Ataque fuerte local vs defensa débil visitante", "success"))
        
        if not recomendaciones:
            recomendaciones.append(("ℹ️ Partido equilibrado", "Analizar en vivo o esperar más info", "info"))
        
        for titulo, desc, tipo in recomendaciones:
            if tipo == "success":
                st.success(f"{titulo} - {desc}")
            else:
                st.info(f"{titulo} - {desc}")
        
        st.caption(f"⏰ Análisis generado: {datetime.now().strftime('%d/%m/%Y %H:%M')} | Powered by Dixon-Coles + Factor L/V + ML v3.0")
    
    else:
        st.error("❌ Error en el análisis. Verifica que los equipos tengan suficientes datos.")

else:
    st.info("👈 Selecciona liga y equipos en el menú lateral para comenzar")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✨ Novedades v3.0")
        st.markdown("""
        - 🏠 **Factor LOCAL/VISITANTE** integrado
        - ✅ **Ajuste automático** de probabilidades
        - ✅ **Tab completo** de rendimiento casa/fuera
        - ✅ **Insights inteligentes** automáticos
        - ✅ **Comparación visual** local vs visitante
        - ✅ **Tarjetas/Corners** por posición (casa/fuera)
        - ✅ **38 ligas mundiales**
        - ✅ **Fuente Inter mejorada**
        """)
    
    with col2:
        st.markdown("### 🚀 Cómo usar")
        st.markdown("""
        1. **Selecciona liga** del menú lateral
        2. **Elige equipos** local y visitante
        3. **Click "ANALIZAR PARTIDO"**
        4. **Explora tabs:**
           - Probabilidades generales
           - Goles por equipo
           - Corners y tarjetas
           - **Rendimiento Local/Visitante** (¡Nuevo!)
           - Gráficos interactivos
        5. **Exporta** si lo necesitas
        
        💡 El factor local/visitante se aplica automáticamente a todas las predicciones.
        """)
