"""
🐾 Animal Shelter Outcome Predictor
Aplicación Streamlit para el refugio Austin Animal Center
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PawPredict · Refugio Animal",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personalizado ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Reset y base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Fondo principal */
.stApp {
    background: #0f1117;
    color: #e8e8e8;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #161b27;
    border-right: 1px solid #1e2535;
}

/* Título principal */
.main-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: #ffffff;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.main-subtitle {
    font-size: 1rem;
    color: #6b7280;
    margin-bottom: 2rem;
    font-weight: 300;
    letter-spacing: 0.05em;
}

/* Cards */
.card {
    background: #161b27;
    border: 1px solid #1e2535;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Resultado principal */
.result-adoption {
    background: linear-gradient(135deg, #052e16 0%, #0d1f12 100%);
    border: 1px solid #16a34a;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-transfer {
    background: linear-gradient(135deg, #0c1a2e 0%, #0d1829 100%);
    border: 1px solid #2563eb;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-return {
    background: linear-gradient(135deg, #1c1003 0%, #1a100a 100%);
    border: 1px solid #d97706;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-atrisk {
    background: linear-gradient(135deg, #1f0505 0%, #1a0808 100%);
    border: 2px solid #dc2626;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    animation: pulse-border 2s infinite;
}
@keyframes pulse-border {
    0%, 100% { box-shadow: 0 0 0 0 rgba(220,38,38,0.3); }
    50%       { box-shadow: 0 0 0 8px rgba(220,38,38,0); }
}

.result-emoji { font-size: 3.5rem; margin-bottom: 0.5rem; }
.result-label {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    font-weight: 400;
    margin-bottom: 0.3rem;
}
.result-prob  { font-size: 3rem; font-weight: 600; }

/* Barra de riesgo */
.risk-bar-container {
    background: #1e2535;
    border-radius: 999px;
    height: 12px;
    width: 100%;
    overflow: hidden;
    margin: 0.5rem 0;
}
.risk-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.8s ease;
}

/* Badges de nivel de riesgo */
.badge-low    { background:#052e16; color:#4ade80; border:1px solid #16a34a; border-radius:999px; padding:4px 14px; font-size:0.8rem; font-weight:600; display:inline-block; }
.badge-medium { background:#1c1003; color:#fbbf24; border:1px solid #d97706; border-radius:999px; padding:4px 14px; font-size:0.8rem; font-weight:600; display:inline-block; }
.badge-high   { background:#1f0505; color:#f87171; border:1px solid #dc2626; border-radius:999px; padding:4px 14px; font-size:0.8rem; font-weight:600; display:inline-block; }

/* Estrategias */
.strategy-card {
    background: #1a0f0f;
    border: 1px solid #7f1d1d;
    border-left: 4px solid #dc2626;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}
.strategy-title { color: #fca5a5; font-weight: 600; margin-bottom: 0.3rem; font-size: 0.95rem; }
.strategy-body  { color: #d1d5db; font-size: 0.88rem; line-height: 1.5; }

/* Prob bars */
.prob-row { display:flex; align-items:center; margin-bottom:0.6rem; gap:0.8rem; }
.prob-label { width: 140px; font-size:0.85rem; color:#9ca3af; flex-shrink:0; }
.prob-bar-bg { flex:1; background:#1e2535; border-radius:999px; height:8px; }
.prob-bar-fg { height:100%; border-radius:999px; }
.prob-val { width:45px; text-align:right; font-size:0.85rem; font-weight:600; color:#e5e7eb; }

/* Historial tabla */
.hist-row { 
    background:#161b27; 
    border:1px solid #1e2535; 
    border-radius:8px; 
    padding:0.7rem 1rem;
    margin-bottom:0.4rem;
    display:flex; align-items:center; gap:1rem;
}

/* Feature impact */
.feat-item { display:flex; align-items:center; gap:0.6rem; margin-bottom:0.5rem; }
.feat-name { font-size:0.82rem; color:#9ca3af; width:160px; flex-shrink:0; }
.feat-bar-bg { flex:1; background:#1e2535; border-radius:999px; height:6px; }
.feat-bar-fg { height:100%; border-radius:999px; background:#dc2626; }
.feat-val { font-size:0.82rem; color:#f87171; width:40px; text-align:right; }

/* Metric cards */
.metric-card {
    background:#161b27;
    border:1px solid #1e2535;
    border-radius:10px;
    padding:1rem;
    text-align:center;
}
.metric-val { font-size:1.8rem; font-weight:700; }
.metric-lbl { font-size:0.78rem; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; }

/* Divider */
.divider { border:none; border-top:1px solid #1e2535; margin:1.5rem 0; }

/* Info page */
.info-section { background:#161b27; border:1px solid #1e2535; border-radius:12px; padding:1.5rem; margin-bottom:1rem; }
.info-title { font-family:'DM Serif Display', serif; font-size:1.3rem; color:#fff; margin-bottom:0.8rem; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background:#161b27; border-radius:8px; padding:4px; gap:4px; }
.stTabs [data-baseweb="tab"] { border-radius:6px; color:#6b7280; }
.stTabs [aria-selected="true"] { background:#1e2535 !important; color:#fff !important; }

/* Inputs */
.stSelectbox > div > div, .stNumberInput > div > div > input {
    background:#1e2535 !important;
    border-color:#2d3748 !important;
    color:#e8e8e8 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #dc2626, #991b1b);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

</style>
""", unsafe_allow_html=True)


# ── Constantes ────────────────────────────────────────────────────────────────
CLASS_ORDER = ["Adoption", "Transfer", "Return to Owner", "At Risk"]

CLASS_CONFIG = {
    "Adoption": {
        "emoji": "🏠", "color": "#4ade80", "label_es": "Adopción",
        "css_class": "result-adoption",
        "msg": "Alta probabilidad de ser adoptado. Continúa con el proceso estándar."
    },
    "Transfer": {
        "emoji": "🚐", "color": "#60a5fa", "label_es": "Traslado",
        "css_class": "result-transfer",
        "msg": "El animal probablemente será trasladado a otro centro asociado."
    },
    "Return to Owner": {
        "emoji": "🔄", "color": "#fbbf24", "label_es": "Devolución",
        "css_class": "result-return",
        "msg": "Es probable que el animal sea devuelto a su propietario."
    },
    "At Risk": {
        "emoji": "⚠️", "color": "#f87171", "label_es": "En Riesgo",
        "css_class": "result-atrisk",
        "msg": "Este animal necesita intervención prioritaria."
    },
}

STRATEGIES = {
    "low": [
        {
            "title": "📋 Monitoreo preventivo",
            "body": "Incluir al animal en el programa de seguimiento semanal. Documentar su estado de salud y comportamiento. Nivel de riesgo bajo — mantener vigilancia estándar."
        },
        {
            "title": "📸 Visibilidad en redes",
            "body": "Publicar perfil del animal en redes sociales y plataformas de adopción para aumentar su exposición y reducir tiempo en el refugio."
        },
    ],
    "medium": [
        {
            "title": "🤝 Programa de acogida temporal",
            "body": "Buscar familia de acogida temporal para reducir el estrés del animal en el refugio. Los animales en acogida tienen 4x más probabilidad de ser adoptados."
        },
        {
            "title": "🏥 Evaluación veterinaria completa",
            "body": "Realizar chequeo exhaustivo para detectar condiciones tratables que puedan estar afectando su adoptabilidad. Documentar y comunicar el estado de salud a potenciales adoptantes."
        },
    ],
    "high": [
        {
            "title": "🚨 Protocolo de intervención urgente",
            "body": "Activar inmediatamente el equipo de rescate de alto riesgo. Contactar organizaciones de rescate especializadas en la raza/especie. El animal necesita salir del refugio en menos de 72 horas."
        },
        {
            "title": "💉 Tratamiento médico / conductual prioritario",
            "body": "Evaluar si condiciones médicas o conductuales son la causa del riesgo. Si son tratables, iniciar tratamiento inmediato y reclasificar. Considerar programa de rehabilitación conductual con voluntarios especializados."
        },
    ],
}

AGE_ORDER = ["Cachorro (<6m)", "Joven (6m-1a)", "Adulto joven (1-3a)", "Adulto (3-7a)", "Senior (>7a)"]

FEATURE_OPTIONS = {
    "AnimalType":      ["Dog", "Cat"],
    "Sex":             ["Male", "Female", "Neutered Male", "Spayed Female", "Intact Male", "Intact Female"],
    "IntakeType":      ["Stray", "Owner Surrender", "Public Assist", "Euthanasia Request", "Abandoned"],
    "IntakeCondition": ["Normal", "Injured", "Sick", "Aged", "Feral", "Pregnant", "Nursing"],
    "AgeGroup":        AGE_ORDER,
    "breed_type":      ["mix", "purebred"],
    "Color_grouped":   ["Monocolor", "Bicolor", "Tricolor"],
    "Season":          ["Primavera", "Verano", "Otoño", "Invierno"],
}

# ── Inicializar session state ─────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ── Funciones auxiliares ───────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    """Carga el modelo y preprocesador. Retorna None si no existen."""
    model, preprocessor, le_target = None, None, None
    try:
        model        = joblib.load("models/best_model.pkl")
        preprocessor = joblib.load("models/preprocessor.pkl")
        le_target    = joblib.load("models/le_target.pkl")
    except Exception:
        pass
    return model, preprocessor, le_target


def age_to_days(age_group: str) -> float:
    mapping = {
        "Cachorro (<6m)":      90,
        "Joven (6m-1a)":       270,
        "Adulto joven (1-3a)": 730,
        "Adulto (3-7a)":       1825,
        "Senior (>7a)":        3285,
    }
    return mapping.get(age_group, 365)


def build_input_df(form_data: dict) -> pd.DataFrame:
    age_days = age_to_days(form_data["AgeGroup"])
    age_log  = np.log1p(age_days)
    return pd.DataFrame([{
        "AgeInDays_log":  age_log,
        "AnimalType":     form_data["AnimalType"],
        "Sex":            form_data["Sex"],
        "IntakeType":     form_data["IntakeType"],
        "IntakeCondition":form_data["IntakeCondition"],
        "AgeGroup":       form_data["AgeGroup"],
        "breed_type":     form_data["breed_type"],
        "Color_grouped":  form_data["Color_grouped"],
        "Season":         form_data["Season"],
    }])


def predict(input_df, model, preprocessor, le_target):
    """Devuelve (clase_predicha, dict_probabilidades)."""
    X_proc      = preprocessor.transform(input_df)
    pred_idx    = model.predict(X_proc)[0]
    proba       = model.predict_proba(X_proc)[0]
    # Mapear índices a nombres de clase
    classes     = le_target.classes_  # orden del LabelEncoder
    proba_dict  = {cls: float(p) for cls, p in zip(classes, proba)}
    clase       = le_target.inverse_transform([pred_idx])[0]
    return clase, proba_dict


def risk_level(at_risk_prob: float):
    if at_risk_prob < 0.25:
        return "low",    "Riesgo Bajo",   "badge-low"
    elif at_risk_prob < 0.55:
        return "medium", "Riesgo Medio",  "badge-medium"
    else:
        return "high",   "Riesgo Alto",   "badge-high"


def get_feature_impacts(form_data: dict) -> list[dict]:
    """Heurística interpretable de qué features impulsan el riesgo."""
    impacts = []
    risk_map = {
        "IntakeCondition": {"Injured": 0.85, "Sick": 0.75, "Aged": 0.65,
                            "Feral": 0.60, "Pregnant": 0.45, "Normal": 0.10},
        "IntakeType":      {"Euthanasia Request": 0.95, "Owner Surrender": 0.55,
                            "Stray": 0.30, "Public Assist": 0.20},
        "AgeGroup":        {"Senior (>7a)": 0.70, "Adulto (3-7a)": 0.30,
                            "Adulto joven (1-3a)": 0.15, "Joven (6m-1a)": 0.10,
                            "Cachorro (<6m)": 0.08},
        "AnimalType":      {"Cat": 0.35, "Dog": 0.20},
        "breed_type":      {"mix": 0.25, "purebred": 0.15},
    }
    labels = {
        "IntakeCondition": "Condición de ingreso",
        "IntakeType":      "Tipo de ingreso",
        "AgeGroup":        "Grupo de edad",
        "AnimalType":      "Tipo de animal",
        "breed_type":      "Tipo de raza",
    }
    for feat, mapping in risk_map.items():
        val   = form_data.get(feat, "")
        score = mapping.get(val, 0.05)
        if score > 0.10:
            impacts.append({"feature": labels[feat], "value": val, "score": score})
    impacts.sort(key=lambda x: x["score"], reverse=True)
    return impacts[:4]


def render_prob_bars(proba_dict: dict):
    color_map = {
        "Adoption": "#4ade80", "Transfer": "#60a5fa",
        "Return to Owner": "#fbbf24", "At Risk": "#f87171",
    }
    html = ""
    for cls in CLASS_ORDER:
        p     = proba_dict.get(cls, 0)
        color = color_map[cls]
        cfg   = CLASS_CONFIG[cls]
        html += f"""
        <div class="prob-row">
            <span class="prob-label">{cfg['emoji']} {cfg['label_es']}</span>
            <div class="prob-bar-bg">
                <div class="prob-bar-fg" style="width:{p*100:.1f}%;background:{color};"></div>
            </div>
            <span class="prob-val">{p*100:.1f}%</span>
        </div>"""
    st.markdown(html, unsafe_allow_html=True)


def render_result_card(clase: str, proba_dict: dict, form_data: dict, animal_name: str = ""):
    cfg      = CLASS_CONFIG[clase]
    prob_cls = proba_dict.get(clase, 0)
    at_risk_p = proba_dict.get("At Risk", 0)

    # Card principal
    name_str = f"<div style='color:#9ca3af;font-size:0.9rem;margin-bottom:0.5rem;'>{animal_name}</div>" if animal_name else ""
    st.markdown(f"""
    <div class="{cfg['css_class']}">
        {name_str}
        <div class="result-emoji">{cfg['emoji']}</div>
        <div class="result-label" style="color:{cfg['color']}">{cfg['label_es']}</div>
        <div class="result-prob" style="color:{cfg['color']}">{prob_cls*100:.1f}%</div>
        <div style="color:#9ca3af;font-size:0.88rem;margin-top:0.5rem;">{cfg['msg']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # Probabilidades de todas las clases
    st.markdown("**Distribución de probabilidades**")
    render_prob_bars(proba_dict)

    # Sección especial At Risk
    if clase == "At Risk" or at_risk_p > 0.15:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        lvl, lvl_label, badge_cls = risk_level(at_risk_p)

        # Nivel de riesgo
        bar_color = {"low": "#4ade80", "medium": "#fbbf24", "high": "#f87171"}[lvl]
        st.markdown(f"""
        <div style="margin-bottom:1rem;">
            <div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.4rem;">
                <span style="color:#9ca3af;font-size:0.85rem;">Nivel de riesgo</span>
                <span class="{badge_cls}">{lvl_label}</span>
            </div>
            <div class="risk-bar-container">
                <div class="risk-bar-fill" style="width:{at_risk_p*100:.1f}%;background:{bar_color};"></div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#4b5563;margin-top:2px;">
                <span>0%</span><span>Probabilidad At Risk: {at_risk_p*100:.1f}%</span><span>100%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Features que impulsan el riesgo
        impacts = get_feature_impacts(form_data)
        if impacts:
            st.markdown("**Factores que aumentan el riesgo**")
            max_s = max(i["score"] for i in impacts)
            html  = ""
            for imp in impacts:
                w = imp["score"] / max_s * 100
                html += f"""
                <div class="feat-item">
                    <span class="feat-name">{imp['feature']}</span>
                    <div class="feat-bar-bg"><div class="feat-bar-fg" style="width:{w:.0f}%"></div></div>
                    <span class="feat-val">{imp['value'][:8]}</span>
                </div>"""
            st.markdown(html, unsafe_allow_html=True)

        # Estrategias de intervención
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        st.markdown("**Estrategias de intervención recomendadas**")
        for strat in STRATEGIES[lvl]:
            st.markdown(f"""
            <div class="strategy-card">
                <div class="strategy-title">{strat['title']}</div>
                <div class="strategy-body">{strat['body']}</div>
            </div>
            """, unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 1.5rem 0;">
        <div style="font-family:'DM Serif Display',serif;font-size:1.6rem;color:#fff;">🐾 PawPredict</div>
        <div style="font-size:0.78rem;color:#4b5563;letter-spacing:0.08em;text-transform:uppercase;">Austin Animal Center</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navegación",
        ["🔮 Predicción individual", "📂 Carga masiva (CSV)", "📋 Historial", "ℹ️ Información"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#1e2535;margin:1.5rem 0;'>", unsafe_allow_html=True)

    # Estado del modelo
    model, preprocessor, le_target = load_artifacts()
    if model is not None:
        st.markdown("""
        <div style="background:#052e16;border:1px solid #16a34a;border-radius:8px;padding:0.7rem 1rem;">
            <div style="color:#4ade80;font-size:0.82rem;font-weight:600;">✅ Modelo cargado</div>
            <div style="color:#6b7280;font-size:0.75rem;margin-top:2px;">Listo para predecir</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#1f0505;border:1px solid #7f1d1d;border-radius:8px;padding:0.7rem 1rem;">
            <div style="color:#f87171;font-size:0.82rem;font-weight:600;">⚠️ Modelo no encontrado</div>
            <div style="color:#6b7280;font-size:0.75rem;margin-top:2px;">Ejecuta el notebook 03 primero</div>
        </div>
        """, unsafe_allow_html=True)

    # Stats del historial
    if st.session_state.history:
        n    = len(st.session_state.history)
        risk = sum(1 for h in st.session_state.history if h["clase"] == "At Risk")
        st.markdown(f"""
        <div style="margin-top:1.5rem;">
            <div style="color:#6b7280;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.6rem;">Esta sesión</div>
            <div style="display:flex;gap:0.8rem;">
                <div class="metric-card" style="flex:1;">
                    <div class="metric-val" style="color:#60a5fa;">{n}</div>
                    <div class="metric-lbl">Evaluados</div>
                </div>
                <div class="metric-card" style="flex:1;">
                    <div class="metric-val" style="color:#f87171;">{risk}</div>
                    <div class="metric-lbl">En riesgo</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — PREDICCIÓN INDIVIDUAL
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔮 Predicción individual":
    st.markdown('<div class="main-title">Predicción individual</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Introduce los datos del animal para obtener su pronóstico de outcome</div>', unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1.1], gap="large")

    with col_form:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Datos del animal**")

        animal_name = st.text_input("Nombre del animal (opcional)", placeholder="Ej: Luna, Max...")

        c1, c2 = st.columns(2)
        with c1:
            animal_type = st.selectbox("Tipo de animal", FEATURE_OPTIONS["AnimalType"])
            intake_type = st.selectbox("Tipo de ingreso", FEATURE_OPTIONS["IntakeType"])
            age_group   = st.selectbox("Grupo de edad", FEATURE_OPTIONS["AgeGroup"])
            breed_type  = st.selectbox("Tipo de raza", FEATURE_OPTIONS["breed_type"])
        with c2:
            sex          = st.selectbox("Sexo", FEATURE_OPTIONS["Sex"])
            intake_cond  = st.selectbox("Condición de ingreso", FEATURE_OPTIONS["IntakeCondition"])
            color_group  = st.selectbox("Color", FEATURE_OPTIONS["Color_grouped"])
            season       = st.selectbox("Estación de ingreso", FEATURE_OPTIONS["Season"])

        st.markdown("</div>", unsafe_allow_html=True)
        predict_btn = st.button("🔮 Predecir outcome", use_container_width=True)

    with col_result:
        form_data = {
            "AnimalType": animal_type, "Sex": sex,
            "IntakeType": intake_type, "IntakeCondition": intake_cond,
            "AgeGroup": age_group, "breed_type": breed_type,
            "Color_grouped": color_group, "Season": season,
        }

        if predict_btn:
            if model is None:
                st.error("⚠️ No se encontró el modelo. Ejecuta primero el notebook 03 y guarda `models/best_model.pkl`, `models/preprocessor.pkl` y `models/le_target.pkl`.")
            else:
                with st.spinner("Analizando..."):
                    input_df         = build_input_df(form_data)
                    clase, proba_dict = predict(input_df, model, preprocessor, le_target)

                # Guardar en historial
                st.session_state.history.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "nombre":    animal_name or "—",
                    "clase":     clase,
                    "prob":      proba_dict.get(clase, 0),
                    "at_risk_p": proba_dict.get("At Risk", 0),
                    "form":      form_data.copy(),
                })

                render_result_card(clase, proba_dict, form_data, animal_name)
        else:
            st.markdown("""
            <div style="height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;
                        border:1px dashed #1e2535;border-radius:16px;padding:3rem;text-align:center;min-height:300px;">
                <div style="font-size:3rem;margin-bottom:1rem;">🐾</div>
                <div style="color:#4b5563;font-size:0.9rem;">Rellena el formulario y pulsa<br><strong style="color:#6b7280;">Predecir outcome</strong></div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — CARGA MASIVA CSV
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📂 Carga masiva (CSV)":
    st.markdown('<div class="main-title">Carga masiva</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Sube un CSV con varios animales para predecir todos a la vez</div>', unsafe_allow_html=True)

    # Plantilla descargable
    template_cols = ["nombre", "AnimalType", "Sex", "IntakeType",
                     "IntakeCondition", "AgeGroup", "breed_type", "Color_grouped", "Season"]
    template_df   = pd.DataFrame([
        ["Luna",  "Dog", "Spayed Female", "Stray",           "Normal",  "Cachorro (<6m)",      "mix",      "Bicolor",   "Primavera"],
        ["Mochi", "Cat", "Intact Male",   "Owner Surrender", "Injured", "Senior (>7a)",         "purebred", "Monocolor", "Invierno"],
        ["Rex",   "Dog", "Intact Male",   "Stray",           "Normal",  "Adulto joven (1-3a)", "mix",      "Tricolor",  "Verano"],
    ], columns=template_cols)

    col_tpl, col_up = st.columns([1, 1], gap="large")
    with col_tpl:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**📥 Plantilla CSV**")
        st.markdown("<div style='color:#6b7280;font-size:0.85rem;margin-bottom:0.8rem;'>Descarga la plantilla, rellénala y súbela.</div>", unsafe_allow_html=True)
        csv_tpl = template_df.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar plantilla", csv_tpl, "plantilla_animales.csv", "text/csv")
        st.dataframe(template_df, use_container_width=True, height=140)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_up:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**📤 Subir CSV**")
        uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

    if uploaded is not None:
        df_up = pd.read_csv(uploaded)
        st.markdown(f"**{len(df_up)} animales cargados** — previsualización:")
        st.dataframe(df_up.head(5), use_container_width=True)

        if st.button("🔮 Predecir todos", use_container_width=True):
            if model is None:
                st.error("Modelo no encontrado.")
            else:
                results = []
                prog    = st.progress(0)
                for i, row in df_up.iterrows():
                    fd = {
                        "AnimalType":     row.get("AnimalType", "Dog"),
                        "Sex":            row.get("Sex", "Male"),
                        "IntakeType":     row.get("IntakeType", "Stray"),
                        "IntakeCondition":row.get("IntakeCondition", "Normal"),
                        "AgeGroup":       row.get("AgeGroup", "Adulto joven (1-3a)"),
                        "breed_type":     row.get("breed_type", "mix"),
                        "Color_grouped":  row.get("Color_grouped", "Monocolor"),
                        "Season":         row.get("Season", "Primavera"),
                    }
                    inp                  = build_input_df(fd)
                    clase, proba_dict    = predict(inp, model, preprocessor, le_target)
                    at_risk_p            = proba_dict.get("At Risk", 0)
                    lvl, lvl_label, _    = risk_level(at_risk_p)
                    results.append({
                        "Nombre":            row.get("nombre", f"Animal {i+1}"),
                        "Predicción":        CLASS_CONFIG[clase]["label_es"],
                        "Confianza (%)":     round(proba_dict.get(clase, 0) * 100, 1),
                        "P(At Risk) (%)":    round(at_risk_p * 100, 1),
                        "Nivel riesgo":      lvl_label,
                        "P(Adoption) (%)":   round(proba_dict.get("Adoption", 0) * 100, 1),
                        "P(Transfer) (%)":   round(proba_dict.get("Transfer", 0) * 100, 1),
                        "P(Return) (%)":     round(proba_dict.get("Return to Owner", 0) * 100, 1),
                    })
                    prog.progress((i + 1) / len(df_up))

                df_res = pd.DataFrame(results)
                st.success(f"✅ {len(df_res)} animales evaluados")

                # Métricas resumen
                n_risk = (df_res["Nivel riesgo"] == "Riesgo Alto").sum()
                n_med  = (df_res["Nivel riesgo"] == "Riesgo Medio").sum()
                n_adop = (df_res["Predicción"] == "Adopción").sum()
                c1, c2, c3 = st.columns(3)
                c1.metric("🏠 Probable adopción", n_adop)
                c2.metric("⚠️ Riesgo medio",       n_med)
                c3.metric("🚨 Riesgo alto",        n_risk)

                st.dataframe(df_res, use_container_width=True)

                # Exportar
                csv_out = df_res.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Descargar resultados CSV", csv_out,
                    f"predicciones_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    "text/csv"
                )

                # Añadir al historial
                for _, row in df_res.iterrows():
                    st.session_state.history.append({
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "nombre":    row["Nombre"],
                        "clase":     next(k for k, v in CLASS_CONFIG.items() if v["label_es"] == row["Predicción"]),
                        "prob":      row["Confianza (%)"] / 100,
                        "at_risk_p": row["P(At Risk) (%)"] / 100,
                        "form":      {},
                    })


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 3 — HISTORIAL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Historial":
    st.markdown('<div class="main-title">Historial de sesión</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Todos los animales evaluados en esta sesión</div>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown("""
        <div style="text-align:center;padding:4rem;color:#4b5563;">
            <div style="font-size:3rem;">📋</div>
            <div style="margin-top:1rem;">Aún no has evaluado ningún animal.<br>Ve a <strong>Predicción individual</strong> o <strong>Carga masiva</strong> para empezar.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Métricas resumen sesión
        hist   = st.session_state.history
        n_tot  = len(hist)
        n_risk = sum(1 for h in hist if h["clase"] == "At Risk")
        n_adop = sum(1 for h in hist if h["clase"] == "Adoption")
        avg_rp = np.mean([h["at_risk_p"] for h in hist]) * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total evaluados",     n_tot)
        c2.metric("🏠 Adopción probable", n_adop)
        c3.metric("🚨 At Risk",           n_risk)
        c4.metric("P(At Risk) media",    f"{avg_rp:.1f}%")

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # Tabla historial
        color_map_cls = {
            "Adoption": "#4ade80", "Transfer": "#60a5fa",
            "Return to Owner": "#fbbf24", "At Risk": "#f87171",
        }
        for h in reversed(hist):
            cfg   = CLASS_CONFIG[h["clase"]]
            color = color_map_cls[h["clase"]]
            lvl, lvl_label, badge = risk_level(h["at_risk_p"])
            st.markdown(f"""
            <div class="hist-row">
                <span style="color:#4b5563;font-size:0.8rem;width:60px;">{h['timestamp']}</span>
                <span style="font-weight:600;color:#e5e7eb;flex:1;">{h['nombre']}</span>
                <span style="color:{color};font-weight:600;">{cfg['emoji']} {cfg['label_es']}</span>
                <span style="color:#9ca3af;font-size:0.85rem;">{h['prob']*100:.1f}%</span>
                <span class="{badge}">{lvl_label}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # Exportar historial
        df_hist = pd.DataFrame([{
            "Hora":        h["timestamp"],
            "Nombre":      h["nombre"],
            "Predicción":  CLASS_CONFIG[h["clase"]]["label_es"],
            "Confianza":   round(h["prob"] * 100, 1),
            "P(At Risk)":  round(h["at_risk_p"] * 100, 1),
        } for h in hist])

        col_exp, col_clr = st.columns([3, 1])
        with col_exp:
            st.download_button(
                "📥 Exportar historial CSV",
                df_hist.to_csv(index=False).encode("utf-8"),
                f"historial_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv",
                use_container_width=True
            )
        with col_clr:
            if st.button("🗑️ Limpiar historial", use_container_width=True):
                st.session_state.history = []
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 4 — INFORMACIÓN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ Información":
    st.markdown('<div class="main-title">Sobre el modelo</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Cómo funciona PawPredict y cómo interpretar los resultados</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="info-section">
            <div class="info-title">🧠 ¿Qué hace el modelo?</div>
            <div style="color:#9ca3af;font-size:0.9rem;line-height:1.7;">
            El modelo analiza las características de cada animal en el momento de su ingreso
            al refugio y predice cuál de los 4 outcomes es más probable:<br><br>
            <strong style="color:#4ade80;">🏠 Adopción</strong> — el animal será adoptado<br>
            <strong style="color:#60a5fa;">🚐 Traslado</strong> — será transferido a otro centro<br>
            <strong style="color:#fbbf24;">🔄 Devolución</strong> — será devuelto a su dueño<br>
            <strong style="color:#f87171;">⚠️ At Risk</strong> — necesita intervención urgente
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-section">
            <div class="info-title">📊 Variables que usa el modelo</div>
            <div style="color:#9ca3af;font-size:0.9rem;line-height:1.8;">
            • <strong style="color:#e5e7eb;">Tipo de animal</strong> — Perro o Gato<br>
            • <strong style="color:#e5e7eb;">Sexo</strong> — incluye si está esterilizado<br>
            • <strong style="color:#e5e7eb;">Tipo de ingreso</strong> — cómo llegó al refugio<br>
            • <strong style="color:#e5e7eb;">Condición de ingreso</strong> — su estado al llegar<br>
            • <strong style="color:#e5e7eb;">Grupo de edad</strong> — de cachorro a senior<br>
            • <strong style="color:#e5e7eb;">Tipo de raza</strong> — mestizo o raza pura<br>
            • <strong style="color:#e5e7eb;">Color</strong> — mono, bi o tricolor<br>
            • <strong style="color:#e5e7eb;">Estación</strong> — época del año del ingreso
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-section">
            <div class="info-title">⚠️ Niveles de riesgo</div>
            <div style="color:#9ca3af;font-size:0.9rem;line-height:1.8;">
            El nivel de riesgo se calcula a partir de la probabilidad que el modelo
            asigna a la clase <em>At Risk</em>:<br><br>
            <span class="badge-low">Riesgo Bajo</span> &nbsp; P(At Risk) &lt; 25% — monitoreo estándar<br><br>
            <span class="badge-medium">Riesgo Medio</span> &nbsp; P(At Risk) 25–55% — intervención preventiva<br><br>
            <span class="badge-high">Riesgo Alto</span> &nbsp; P(At Risk) &gt; 55% — acción urgente
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-section">
            <div class="info-title">⚙️ Detalles técnicos</div>
            <div style="color:#9ca3af;font-size:0.9rem;line-height:1.8;">
            • Entrenado con <strong style="color:#e5e7eb;">110.000+ registros</strong> del Austin Animal Center<br>
            • Clasificación multiclase con <strong style="color:#e5e7eb;">4 outcomes</strong><br>
            • Balanceo de clases con <strong style="color:#e5e7eb;">SMOTE</strong> (At Risk = 3.7%)<br>
            • Métrica principal: <strong style="color:#e5e7eb;">F1-macro</strong><br>
            • Objetivo mínimo: <strong style="color:#e5e7eb;">Recall At Risk ≥ 0.55</strong><br>
            • Preprocesado: <strong style="color:#e5e7eb;">RobustScaler + OHE + OrdinalEncoder</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-section" style="border-color:#7f1d1d;">
        <div class="info-title" style="color:#fca5a5;">⚠️ Limitaciones importantes</div>
        <div style="color:#9ca3af;font-size:0.9rem;line-height:1.7;">
        Este modelo es una herramienta de <strong style="color:#e5e7eb;">apoyo a la decisión</strong>, no un sustituto del criterio veterinario o del personal del refugio.
        Las predicciones se basan en patrones históricos y pueden no reflejar la situación individual de cada animal.
        Úsalas como punto de partida para priorizar recursos, no como sentencia definitiva.
        </div>
    </div>
    """, unsafe_allow_html=True)