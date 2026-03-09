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
from supabase import create_client

# Fix compatibilidad LargeUtf8 con versiones antiguas de Streamlit
try:
    pd.options.future.infer_string = False
except Exception:
    pass

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PawPredict · Refugio Animal",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Conexión Supabase ─────────────────────────────
@st.cache_resource
def get_supabase():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_ANON_KEY"]
    return create_client(url, key)

supabase = get_supabase()
# st.write("Supabase conectado:", supabase is not None)


# ── CSS personalizado ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: #0f1117; color: #e8e8e8; }

[data-testid="stSidebar"] {
    background: #161b27;
    border-right: 1px solid #1e2535;
}

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

.card {
    background: #161b27;
    border: 1px solid #1e2535;
    border-radius: 12px;
    #padding: 1.5rem;
    margin-bottom: 1rem;
}

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
.result-prob { font-size: 3rem; font-weight: 600; }

.risk-bar-container {
    background: #1e2535;
    border-radius: 999px;
    height: 12px;
    width: 100%;
    overflow: hidden;
    margin: 0.5rem 0;
}
.risk-bar-fill { height: 100%; border-radius: 999px; transition: width 0.8s ease; }

.badge-low    { background:#052e16; color:#4ade80; border:1px solid #16a34a; border-radius:999px; padding:4px 14px; font-size:0.8rem; font-weight:600; display:inline-block; }
.badge-medium { background:#1c1003; color:#fbbf24; border:1px solid #d97706; border-radius:999px; padding:4px 14px; font-size:0.8rem; font-weight:600; display:inline-block; }
.badge-high   { background:#1f0505; color:#f87171; border:1px solid #dc2626; border-radius:999px; padding:4px 14px; font-size:0.8rem; font-weight:600; display:inline-block; }

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

.prob-row { display:flex; align-items:center; margin-bottom:0.6rem; gap:0.8rem; }
.prob-label { width: 140px; font-size:0.85rem; color:#9ca3af; flex-shrink:0; }
.prob-bar-bg { flex:1; background:#1e2535; border-radius:999px; height:8px; }
.prob-bar-fg { height:100%; border-radius:999px; }
.prob-val { width:45px; text-align:right; font-size:0.85rem; font-weight:600; color:#e5e7eb; }

.hist-row {
    background:#161b27;
    border:1px solid #1e2535;
    border-radius:8px;
    padding:0.7rem 1rem;
    margin-bottom:0.4rem;
    display:flex; align-items:center; gap:1rem;
}

.feat-item { display:flex; align-items:center; gap:0.6rem; margin-bottom:0.5rem; }
.feat-name { font-size:0.82rem; color:#9ca3af; width:160px; flex-shrink:0; }
.feat-bar-bg { flex:1; background:#1e2535; border-radius:999px; height:6px; }
.feat-bar-fg { height:100%; border-radius:999px; background:#dc2626; }
.feat-val { font-size:0.82rem; color:#f87171; width:40px; text-align:right; }

.metric-card {
    background:#161b27;
    border:1px solid #1e2535;
    border-radius:10px;
    padding:1rem;
    text-align:center;
}
.metric-val { font-size:1.8rem; font-weight:700; }
.metric-lbl { font-size:0.78rem; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; }

.divider { border:none; border-top:1px solid #1e2535; margin:1.5rem 0; }

.info-section { background:#161b27; border:1px solid #1e2535; border-radius:12px; padding:1.5rem; margin-bottom:1rem; }
.info-title { font-family:'DM Serif Display', serif; font-size:1.3rem; color:#fff; margin-bottom:0.8rem; }

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
    "IntakeType":      ["Stray", "Owner Surrender", "Public Assist", "Euthanasia Request",  "Abandoned"],
    "IntakeCondition": ["Normal", "Injured", "Sick", "Aged", "Feral", "Pregnant", "Nursing"],
    "AgeGroup":        AGE_ORDER,
    "breed_type":      ["mix", "purebred"],
    "Color_grouped":   ["Monocolor", "Bicolor", "Tricolor"],
    "Season":          ["Primavera", "Verano", "Otoño", "Invierno"],
}

FORM_DEFAULTS = {
    "form_nombre":    "",
    "form_animal":    "Dog",
    "form_sex":       "Male",
    "form_intake":    "Stray",
    "form_condition": "Normal",
    "form_age":       "Adulto joven (1-3a)",
    "form_breed":     "mix",
    "form_color":     "Monocolor",
    "form_season":    "Primavera",
}

FORM_KEY_MAP = {
    "AnimalType":      "form_animal",
    "Sex":             "form_sex",
    "IntakeType":      "form_intake",
    "IntakeCondition": "form_condition",
    "AgeGroup":        "form_age",
    "breed_type":      "form_breed",
    "Color_grouped":   "form_color",
    "Season":          "form_season",
}

# ── Inicializar session state ─────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "nav_page" not in st.session_state:
    st.session_state.nav_page = None
for k, v in FORM_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Funciones auxiliares ───────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    """Carga el modelo, preprocesador, le_target y umbral óptimo."""
    model, preprocessor, le_target, umbral = None, None, None, 0.5
    try:
        model        = joblib.load("models/xgboost_optimizado.pkl")
        preprocessor = joblib.load("models/preprocessor.pkl")
        le_target    = joblib.load("models/le_target.pkl")
    except Exception:
        pass
    try:
        umbral = float(joblib.load("models/umbral_optimo.pkl"))
    except Exception:
        umbral = 0.5
    return model, preprocessor, le_target, umbral


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
        "AgeInDays_log":   age_log,
        "AnimalType":      form_data["AnimalType"],
        "Sex":             form_data["Sex"],
        "IntakeType":      form_data["IntakeType"],
        "IntakeCondition": form_data["IntakeCondition"],
        "AgeGroup":        form_data["AgeGroup"],
        "breed_type":      form_data["breed_type"],
        "Color_grouped":   form_data["Color_grouped"],
        "Season":          form_data["Season"],
    }])


def predict(input_df, model, preprocessor, le_target, umbral=0.5):
    """Devuelve (clase_predicha, dict_probabilidades). Aplica umbral óptimo para At Risk."""
    X_proc     = preprocessor.transform(input_df)
    proba      = model.predict_proba(X_proc)[0]
    classes    = le_target.classes_
    proba_dict = {cls: float(p) for cls, p in zip(classes, proba)}

    at_risk_prob = proba_dict.get("At Risk", 0.0)
    if at_risk_prob >= umbral:
        clase = "At Risk"
    else:
        pred_idx = int(proba.argmax())
        clase    = le_target.inverse_transform([pred_idx])[0]

    return clase, proba_dict


def risk_level(at_risk_prob: float):
    if at_risk_prob < 0.25:
        return "low",    "Riesgo Bajo",  "badge-low"
    elif at_risk_prob < 0.55:
        return "medium", "Riesgo Medio", "badge-medium"
    else:
        return "high",   "Riesgo Alto",  "badge-high"


def get_feature_impacts(form_data: dict):
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
        html += (
            f"<div class='prob-row'>"
            f"<span class='prob-label'>{cfg['emoji']} {cfg['label_es']}</span>"
            f"<div class='prob-bar-bg'><div class='prob-bar-fg' style='width:{p*100:.1f}%;background:{color};'></div></div>"
            f"<span class='prob-val'>{p*100:.1f}%</span>"
            f"</div>"
        )
    st.markdown(html, unsafe_allow_html=True)


def render_result_card(clase: str, proba_dict: dict, form_data: dict, animal_name: str = ""):
    cfg       = CLASS_CONFIG[clase]
    prob_cls  = proba_dict.get(clase, 0)
    at_risk_p = proba_dict.get("At Risk", 0)

    name_html = f"<div style='color:#9ca3af;font-size:0.9rem;margin-bottom:0.5rem;'>{animal_name}</div>" if animal_name else ""
    card_html = (
        f"<div class='{cfg['css_class']}'>"
        f"{name_html}"
        f"<div class='result-emoji'>{cfg['emoji']}</div>"
        f"<div class='result-label' style='color:{cfg['color']}'>{cfg['label_es']}</div>"
        f"<div class='result-prob' style='color:{cfg['color']}'>{prob_cls*100:.1f}%</div>"
        f"<div style='color:#9ca3af;font-size:0.88rem;margin-top:0.5rem;'>{cfg['msg']}</div>"
        f"</div>"
    )
    st.markdown(card_html, unsafe_allow_html=True)
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    st.markdown("**Distribución de probabilidades**")
    render_prob_bars(proba_dict)

    if clase == "At Risk" or at_risk_p > 0.15:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        lvl, lvl_label, badge_cls = risk_level(at_risk_p)
        bar_color = {"low": "#4ade80", "medium": "#fbbf24", "high": "#f87171"}[lvl]

        risk_html = (
            f"<div style='margin-bottom:1rem;'>"
            f"<div style='display:flex;align-items:center;gap:0.8rem;margin-bottom:0.4rem;'>"
            f"<span style='color:#9ca3af;font-size:0.85rem;'>Nivel de riesgo</span>"
            f"<span class='{badge_cls}'>{lvl_label}</span>"
            f"</div>"
            f"<div class='risk-bar-container'>"
            f"<div class='risk-bar-fill' style='width:{at_risk_p*100:.1f}%;background:{bar_color};'></div>"
            f"</div>"
            f"<div style='display:flex;justify-content:space-between;font-size:0.75rem;color:#4b5563;margin-top:2px;'>"
            f"<span>0%</span><span>Probabilidad At Risk: {at_risk_p*100:.1f}%</span><span>100%</span>"
            f"</div></div>"
        )
        st.markdown(risk_html, unsafe_allow_html=True)

        impacts = get_feature_impacts(form_data)
        if impacts:
            st.markdown("**Factores que aumentan el riesgo**")
            max_s = max(i["score"] for i in impacts)
            html  = ""
            for imp in impacts:
                w     = imp["score"] / max_s * 100
                html += (
                    f"<div class='feat-item'>"
                    f"<span class='feat-name'>{imp['feature']}</span>"
                    f"<div class='feat-bar-bg'><div class='feat-bar-fg' style='width:{w:.0f}%'></div></div>"
                    f"<span class='feat-val'>{imp['value'][:8]}</span>"
                    f"</div>"
                )
            st.markdown(html, unsafe_allow_html=True)

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        st.markdown("**Estrategias de intervención recomendadas**")
        for strat in STRATEGIES[lvl]:
            strat_html = (
                f"<div class='strategy-card'>"
                f"<div class='strategy-title'>{strat['title']}</div>"
                f"<div class='strategy-body'>{strat['body']}</div>"
                f"</div>"
            )
            st.markdown(strat_html, unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='padding:1rem 0 1.5rem 0;'>"
        "<div style='font-family:DM Serif Display,serif;font-size:1.6rem;color:#fff;'>🐾 PawPredict</div>"
        "<div style='font-size:0.78rem;color:#4b5563;letter-spacing:0.08em;text-transform:uppercase;'>Austin Animal Center</div>"
        "</div>",
        unsafe_allow_html=True
    )

    # Navegación programática desde historial
    if st.session_state.get("nav_page"):
        default_page = st.session_state["nav_page"]
        st.session_state["nav_page"] = None
        st.session_state["radio_key"] = st.session_state.get("radio_key", 0) + 1
    else:
        default_page = "🔮 Predicción individual"

    PAGES = ["🔮 Predicción individual", "📂 Carga masiva (CSV)", "📋 Historial", "📊 Feedback", "ℹ️ Información"]

    page = st.radio(
        "Navegación",
        PAGES,
        index=PAGES.index(default_page),
        key=f"nav_radio_{st.session_state.get('radio_key', 0)}",
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#1e2535;margin:1.5rem 0;'>", unsafe_allow_html=True)

    model, preprocessor, le_target, umbral = load_artifacts()
    if model is not None:
        st.markdown(
            "<div style='background:#052e16;border:1px solid #16a34a;border-radius:8px;padding:0.7rem 1rem;'>"
            "<div style='color:#4ade80;font-size:0.82rem;font-weight:600;'>✅ Modelo cargado</div>"
            "<div style='color:#6b7280;font-size:0.75rem;margin-top:2px;'>XGBoost · Optuna optimizado</div>"
            "</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div style='color:#4b5563;font-size:0.75rem;margin-top:0.4rem;'>"
            f"Umbral At Risk: <strong style='color:#f87171;'>{umbral:.2f}</strong></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='background:#1f0505;border:1px solid #7f1d1d;border-radius:8px;padding:0.7rem 1rem;'>"
            "<div style='color:#f87171;font-size:0.82rem;font-weight:600;'>⚠️ Modelo no encontrado</div>"
            "<div style='color:#6b7280;font-size:0.75rem;margin-top:2px;'>Ejecuta el notebook 04 primero</div>"
            "</div>",
            unsafe_allow_html=True
        )

    if st.session_state.history:
        n    = len(st.session_state.history)
        risk = sum(1 for h in st.session_state.history if h["clase"] == "At Risk")
        st.markdown(
            f"<div style='margin-top:1.5rem;'>"
            f"<div style='color:#6b7280;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.6rem;'>Esta sesión</div>"
            f"<div style='display:flex;gap:0.8rem;'>"
            f"<div class='metric-card' style='flex:1;'><div class='metric-val' style='color:#60a5fa;'>{n}</div><div class='metric-lbl'>Evaluados</div></div>"
            f"<div class='metric-card' style='flex:1;'><div class='metric-val' style='color:#f87171;'>{risk}</div><div class='metric-lbl'>En riesgo</div></div>"
            f"</div></div>",
            unsafe_allow_html=True
        )


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — PREDICCIÓN INDIVIDUAL
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔮 Predicción individual":
    st.markdown('<div class="main-title">Predicción individual</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Introduce los datos del animal para obtener su pronóstico de outcome</div>', unsafe_allow_html=True)

    FORM_DEFAULTS = {
        "val_nombre":    "",
        "val_animal":    "Dog",
        "val_sex":       "Male",
        "val_intake":    "Stray",
        "val_condition": "Normal",
        "val_age":       "Adulto joven (1-3a)",
        "val_breed":     "mix",
        "val_color":     "Monocolor",
        "val_season":    "Primavera",
    }
    for k, v in FORM_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

    col_form, col_result = st.columns([1, 1.1], gap="large")

    with col_form:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Datos del animal**")

        # Sincronizar nombre solo cuando viene del historial (reabrir)
        if st.session_state.get("restore_nombre"):
            st.session_state["form_nombre"] = st.session_state["val_nombre"]
            st.session_state["restore_nombre"] = False

        animal_name = st.text_input(
            "Nombre del animal (opcional)",
            placeholder="Ej: Luna, Max...",
            #value=st.session_state["val_nombre"],
            key="form_nombre"
        )
        st.session_state["val_nombre"] = animal_name

        c1, c2 = st.columns(2)
        with c1:
            animal_type = st.selectbox("Tipo de animal",       FEATURE_OPTIONS["AnimalType"],
                index=FEATURE_OPTIONS["AnimalType"].index(st.session_state["val_animal"]),
                key="form_animal")
            st.session_state["val_animal"] = animal_type

            intake_type = st.selectbox("Tipo de ingreso",      FEATURE_OPTIONS["IntakeType"],
                index=FEATURE_OPTIONS["IntakeType"].index(st.session_state["val_intake"]),
                key="form_intake")
            st.session_state["val_intake"] = intake_type

            age_group   = st.selectbox("Grupo de edad",        FEATURE_OPTIONS["AgeGroup"],
                index=FEATURE_OPTIONS["AgeGroup"].index(st.session_state["val_age"]),
                key="form_age")
            st.session_state["val_age"] = age_group

            breed_type  = st.selectbox("Tipo de raza",         FEATURE_OPTIONS["breed_type"],
                index=FEATURE_OPTIONS["breed_type"].index(st.session_state["val_breed"]),
                key="form_breed")
            st.session_state["val_breed"] = breed_type

        with c2:
            sex         = st.selectbox("Sexo",                 FEATURE_OPTIONS["Sex"],
                index=FEATURE_OPTIONS["Sex"].index(st.session_state["val_sex"]),
                key="form_sex")
            st.session_state["val_sex"] = sex

            intake_cond = st.selectbox("Condición de ingreso", FEATURE_OPTIONS["IntakeCondition"],
                index=FEATURE_OPTIONS["IntakeCondition"].index(st.session_state["val_condition"]),
                key="form_condition")
            st.session_state["val_condition"] = intake_cond

            color_group = st.selectbox("Color",                FEATURE_OPTIONS["Color_grouped"],
                index=FEATURE_OPTIONS["Color_grouped"].index(st.session_state["val_color"]),
                key="form_color")
            st.session_state["val_color"] = color_group

            season      = st.selectbox("Estación de ingreso",  FEATURE_OPTIONS["Season"],
                index=FEATURE_OPTIONS["Season"].index(st.session_state["val_season"]),
                key="form_season")
            st.session_state["val_season"] = season

        st.markdown("</div>", unsafe_allow_html=True)

        predict_btn = st.button("🔮 Predecir outcome", use_container_width=True)
        if st.button("🗑️ Limpiar formulario", use_container_width=True):
            for k, v in FORM_DEFAULTS.items():
                st.session_state[k] = v
            st.session_state["restore_nombre"] = True
            st.experimental_rerun()

    with col_result:
        form_data = {
            "AnimalType":     animal_type,  "Sex":             sex,
            "IntakeType":     intake_type,  "IntakeCondition": intake_cond,
            "AgeGroup":       age_group,    "breed_type":      breed_type,
            "Color_grouped":  color_group,  "Season":          season,
        }

        if predict_btn:
            if model is None:
                st.error("⚠️ No se encontró el modelo. Ejecuta primero el notebook 04.")
            else:
                with st.spinner("Analizando..."):
                    input_df          = build_input_df(form_data)
                    clase, proba_dict = predict(input_df, model, preprocessor, le_target, umbral)

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
            st.markdown(
                "<div style='height:100%;display:flex;flex-direction:column;align-items:center;"
                "justify-content:center;border:1px dashed #1e2535;border-radius:16px;padding:3rem;"
                "text-align:center;min-height:300px;'>"
                "<div style='font-size:3rem;margin-bottom:1rem;'>🐾</div>"
                "<div style='color:#4b5563;font-size:0.9rem;'>Rellena el formulario y pulsa"
                "<br><strong style='color:#6b7280;'>Predecir outcome</strong></div>"
                "</div>",
                unsafe_allow_html=True
            )


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — CARGA MASIVA CSV
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📂 Carga masiva (CSV)":
    st.markdown('<div class="main-title">Carga masiva</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Sube un CSV con varios animales para predecir todos a la vez</div>', unsafe_allow_html=True)

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
        st.dataframe(template_df.astype(str), use_container_width=True, height=140)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_up:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**📤 Subir CSV**")
        uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

    if uploaded is not None:
        df_up = pd.read_csv(uploaded).astype(str)
        st.markdown(f"**{len(df_up)} animales cargados** — previsualización:")
        st.dataframe(df_up.head(5).astype(str), use_container_width=True)

        if st.button("🔮 Predecir todos", use_container_width=True):
            if model is None:
                st.error("Modelo no encontrado.")
            else:
                results = []
                prog    = st.progress(0)
                for i, row in df_up.iterrows():
                    fd = {
                        "AnimalType":     row.get("AnimalType",     "Dog"),
                        "Sex":            row.get("Sex",            "Male"),
                        "IntakeType":     row.get("IntakeType",     "Stray"),
                        "IntakeCondition":row.get("IntakeCondition","Normal"),
                        "AgeGroup":       row.get("AgeGroup",       "Adulto joven (1-3a)"),
                        "breed_type":     row.get("breed_type",     "mix"),
                        "Color_grouped":  row.get("Color_grouped",  "Monocolor"),
                        "Season":         row.get("Season",         "Primavera"),
                    }
                    inp               = build_input_df(fd)
                    clase, proba_dict = predict(inp, model, preprocessor, le_target, umbral)
                    at_risk_p         = proba_dict.get("At Risk", 0)
                    lvl, lvl_label, _ = risk_level(at_risk_p)
                    results.append({
                        "Nombre":          row.get("nombre", f"Animal {i+1}"),
                        "Predicción":      CLASS_CONFIG[clase]["label_es"],
                        "Confianza (%)":   round(proba_dict.get(clase, 0) * 100, 1),
                        "P(At Risk) (%)":  round(at_risk_p * 100, 1),
                        "Nivel riesgo":    lvl_label,
                        "P(Adoption) (%)": round(proba_dict.get("Adoption", 0) * 100, 1),
                        "P(Transfer) (%)": round(proba_dict.get("Transfer", 0) * 100, 1),
                        "P(Return) (%)":   round(proba_dict.get("Return to Owner", 0) * 100, 1),
                    })
                    prog.progress((i + 1) / len(df_up))

                df_res = pd.DataFrame(results)
                st.success(f"✅ {len(df_res)} animales evaluados")

                n_risk = (df_res["Nivel riesgo"] == "Riesgo Alto").sum()
                n_med  = (df_res["Nivel riesgo"] == "Riesgo Medio").sum()
                n_adop = (df_res["Predicción"] == "Adopción").sum()
                c1, c2, c3 = st.columns(3)
                c1.metric("🏠 Probable adopción", n_adop)
                c2.metric("⚠️ Riesgo medio",      n_med)
                c3.metric("🚨 Riesgo alto",       n_risk)

                st.dataframe(df_res.astype(str), use_container_width=True)

                csv_out = df_res.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Descargar resultados CSV", csv_out,
                    f"predicciones_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    "text/csv"
                )

                for _, row in df_res.iterrows():
                    st.session_state.history.append({
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "nombre":    row["Nombre"],
                        "clase":     next(k for k, v in CLASS_CONFIG.items() if v["label_es"] == row["Predicción"]),
                        "prob":      float(row["Confianza (%)"]) / 100,
                        "at_risk_p": float(row["P(At Risk) (%)"]) / 100,
                        "form":      {},
                    })


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 3 — HISTORIAL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Historial":
    st.markdown('<div class="main-title">Historial de sesión</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Todos los animales evaluados en esta sesión</div>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown(
            "<div style='text-align:center;padding:4rem;color:#4b5563;'>"
            "<div style='font-size:3rem;'>📋</div>"
            "<div style='margin-top:1rem;'>Aún no has evaluado ningún animal.<br>"
            "Ve a <strong>Predicción individual</strong> o <strong>Carga masiva</strong> para empezar.</div>"
            "</div>",
            unsafe_allow_html=True
        )
    else:
        hist   = st.session_state.history
        n_tot  = len(hist)
        n_risk = sum(1 for h in hist if h["clase"] == "At Risk")
        n_adop = sum(1 for h in hist if h["clase"] == "Adoption")
        avg_rp = np.mean([h["at_risk_p"] for h in hist]) * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total evaluados",      n_tot)
        c2.metric("🏠 Adopción probable", n_adop)
        c3.metric("🚨 At Risk",           n_risk)
        c4.metric("P(At Risk) media",     f"{avg_rp:.1f}%")

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        color_map_cls = {
            "Adoption": "#4ade80", "Transfer": "#60a5fa",
            "Return to Owner": "#fbbf24", "At Risk": "#f87171",
        }

        for idx, h in enumerate(reversed(hist)):
            cfg   = CLASS_CONFIG[h["clase"]]
            color = color_map_cls[h["clase"]]
            lvl, lvl_label, badge = risk_level(h["at_risk_p"])

            col_info, col_btn = st.columns([5, 1])
            with col_info:
                st.markdown(
                    f"<div class='hist-row'>"
                    f"<span style='color:#4b5563;font-size:0.8rem;width:60px;'>{h['timestamp']}</span>"
                    f"<span style='font-weight:600;color:#e5e7eb;flex:1;'>{h['nombre']}</span>"
                    f"<span style='color:{color};font-weight:600;'>{cfg['emoji']} {cfg['label_es']}</span>"
                    f"<span style='color:#9ca3af;font-size:0.85rem;'>{h['prob']*100:.1f}%</span>"
                    f"<span class='{badge}'>{lvl_label}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            with col_btn:
                if h.get("form") and st.button("↩ Reabrir", key=f"reopen_{idx}", use_container_width=True):
                    st.session_state["val_nombre"] = h["nombre"] if h["nombre"] != "—" else ""
                    st.session_state["restore_nombre"] = True
                    st.session_state["val_animal"]    = h["form"].get("AnimalType",     "Dog")
                    st.session_state["val_sex"]       = h["form"].get("Sex",            "Male")
                    st.session_state["val_intake"]    = h["form"].get("IntakeType",     "Stray")
                    st.session_state["val_condition"] = h["form"].get("IntakeCondition","Normal")
                    st.session_state["val_age"]       = h["form"].get("AgeGroup",       "Adulto joven (1-3a)")
                    st.session_state["val_breed"]     = h["form"].get("breed_type",     "mix")
                    st.session_state["val_color"]     = h["form"].get("Color_grouped",  "Monocolor")
                    st.session_state["val_season"]    = h["form"].get("Season",         "Primavera")
                    st.session_state["nav_page"]      = "🔮 Predicción individual"
                    st.experimental_rerun()

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        df_hist = pd.DataFrame([{
            "Hora":       h["timestamp"],
            "Nombre":     h["nombre"],
            "Predicción": CLASS_CONFIG[h["clase"]]["label_es"],
            "Confianza":  round(h["prob"] * 100, 1),
            "P(At Risk)": round(h["at_risk_p"] * 100, 1),
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
                st.experimental_rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 4 — FEEDBACK
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Feedback":
    import json

    FEEDBACK_PATH = "data/feedback.csv"
    CLASSES_ES    = ["Adopción", "Transferencia", "Retorno al dueño", "En Riesgo"]
    CLASSES_EN    = ["Adoption", "Transfer", "Return to Owner", "At Risk"]
    ES_TO_EN      = dict(zip(CLASSES_ES, CLASSES_EN))
    EN_TO_ES      = dict(zip(CLASSES_EN, CLASSES_ES))

    # Inicializar feedback en session_state
    if "feedback_list" not in st.session_state:
        # Cargar desde disco si existe
        if os.path.exists(FEEDBACK_PATH):
            try:
                df_fb_load = pd.read_csv(FEEDBACK_PATH)
                st.session_state.feedback_list = df_fb_load.to_dict("records")
            except Exception:
                st.session_state.feedback_list = []
        else:
            st.session_state.feedback_list = []

    st.markdown('<div class="main-title">Feedback del modelo</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Registra el outcome real para monitorizar la performance en producción</div>', unsafe_allow_html=True)

    tab_form, tab_metrics = st.tabs(["📝 Registrar feedback", "📈 Métricas en tiempo real"])

    # ── TAB 1: FORMULARIO ─────────────────────────────────────────────────────
    with tab_form:
        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2, gap="large")

        # ── OPCIÓN A: desde historial ──────────────────────────────────────
        with col_a:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**📋 Desde el historial**")
            st.markdown("<div style='color:#6b7280;font-size:0.85rem;margin-bottom:0.8rem;'>Selecciona un animal ya evaluado y confirma su outcome real.</div>", unsafe_allow_html=True)

            if not st.session_state.history:
                st.markdown("<div style='color:#4b5563;font-size:0.85rem;'>Aún no hay animales en el historial.</div>", unsafe_allow_html=True)
            else:
                hist_options = {
                    f"{h['timestamp']} — {h['nombre']} ({EN_TO_ES.get(h['clase'], h['clase'])})": i
                    for i, h in enumerate(st.session_state.history)
                }
                selected_label = st.selectbox("Animal evaluado", list(hist_options.keys()), key="fb_hist_sel")
                selected_idx   = hist_options[selected_label]
                selected_h     = st.session_state.history[selected_idx]

                pred_es = EN_TO_ES.get(selected_h["clase"], selected_h["clase"])
                st.markdown(
                    f"<div style='background:#0f172a;border-radius:8px;padding:0.6rem 1rem;margin:0.5rem 0;"
                    f"font-size:0.85rem;color:#9ca3af;'>Predicción: "
                    f"<strong style='color:#e5e7eb;'>{pred_es}</strong> "
                    f"({selected_h['prob']*100:.1f}%)</div>",
                    unsafe_allow_html=True
                )

                real_from_hist = st.selectbox("Outcome real", CLASSES_ES, key="fb_hist_real")
                notes_hist     = st.text_input("Notas (opcional)", placeholder="Ej: adoptado por familia el día siguiente", key="fb_hist_notes")

                if st.button("✅ Guardar feedback", use_container_width=True, key="fb_hist_btn"):
                    entry = {
                        "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "nombre":      selected_h["nombre"],
                        "prediccion":  selected_h["clase"],
                        "real":        ES_TO_EN[real_from_hist],
                        "correcto":    selected_h["clase"] == ES_TO_EN[real_from_hist],
                        "at_risk_p":   round(selected_h["at_risk_p"] * 100, 1),
                        "notas":       notes_hist,
                        "fuente":      "historial",
                    }
                    st.session_state.feedback_list.append(entry)
                    # Guardar en disco
                    os.makedirs("data", exist_ok=True)
                    pd.DataFrame(st.session_state.feedback_list).to_csv(FEEDBACK_PATH, index=False)
                    st.success(f"✅ Feedback guardado — {'✓ Correcto' if entry['correcto'] else '✗ Incorrecto'}")

            st.markdown("</div>", unsafe_allow_html=True)

        # ── OPCIÓN B: formulario libre ─────────────────────────────────────
        with col_b:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**✏️ Formulario libre**")
            st.markdown("<div style='color:#6b7280;font-size:0.85rem;margin-bottom:0.8rem;'>Introduce manualmente los datos de cualquier animal.</div>", unsafe_allow_html=True)

            nombre_free  = st.text_input("Nombre del animal", placeholder="Ej: Luna", key="fb_free_nombre")
            pred_free    = st.selectbox("Predicción del modelo", CLASSES_ES, key="fb_free_pred")
            real_free    = st.selectbox("Outcome real",          CLASSES_ES, key="fb_free_real")
            at_risk_free = st.slider("P(At Risk) del modelo (%)", 0, 100, 0, key="fb_free_atrisk")
            notes_free   = st.text_input("Notas (opcional)", placeholder="Ej: animal enfermo al ingreso", key="fb_free_notes")

            if st.button("✅ Guardar feedback", use_container_width=True, key="fb_free_btn"):
                if not nombre_free.strip():
                    st.warning("Introduce un nombre para identificar al animal.")
                else:
                    entry = {
                        "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "nombre":      nombre_free.strip(),
                        "prediccion":  ES_TO_EN[pred_free],
                        "real":        ES_TO_EN[real_free],
                        "correcto":    pred_free == real_free,
                        "at_risk_p":   at_risk_free,
                        "notas":       notes_free,
                        "fuente":      "manual",
                    }
                    st.session_state.feedback_list.append(entry)
                    os.makedirs("data", exist_ok=True)
                    pd.DataFrame(st.session_state.feedback_list).to_csv(FEEDBACK_PATH, index=False)
                    st.success(f"✅ Feedback guardado — {'✓ Correcto' if entry['correcto'] else '✗ Incorrecto'}")

            st.markdown("</div>", unsafe_allow_html=True)

        # ── TABLA FEEDBACK REGISTRADO ──────────────────────────────────────
        if st.session_state.feedback_list:
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            st.markdown("**Feedback registrado en esta sesión**")
            df_fb_show = pd.DataFrame([{
                "Hora":        f["timestamp"][-8:],
                "Nombre":      f["nombre"],
                "Predicción":  EN_TO_ES.get(f["prediccion"], f["prediccion"]),
                "Real":        EN_TO_ES.get(f["real"], f["real"]),
                "✓/✗":        "✓" if f["correcto"] else "✗",
                "P(At Risk)":  f"{f['at_risk_p']}%",
                "Fuente":      f["fuente"],
            } for f in st.session_state.feedback_list])
            st.dataframe(df_fb_show.astype(str), use_container_width=True)

            col_dl, col_rm = st.columns([3, 1])
            with col_dl:
                st.download_button(
                    "📥 Exportar feedback CSV",
                    pd.DataFrame(st.session_state.feedback_list).to_csv(index=False).encode("utf-8"),
                    f"feedback_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    "text/csv",
                    use_container_width=True
                )
            with col_rm:
                if st.button("🗑️ Limpiar feedback", use_container_width=True):
                    st.session_state.feedback_list = []
                    if os.path.exists(FEEDBACK_PATH):
                        os.remove(FEEDBACK_PATH)
                    st.experimental_rerun()

    # ── TAB 2: MÉTRICAS ───────────────────────────────────────────────────────
    with tab_metrics:
        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

        fb = st.session_state.feedback_list
        if len(fb) < 2:
            st.markdown(
                "<div style='text-align:center;padding:3rem;color:#4b5563;'>"
                "<div style='font-size:3rem;'>📊</div>"
                "<div style='margin-top:1rem;'>Necesitas al menos 2 registros de feedback<br>para ver las métricas.</div>"
                "</div>",
                unsafe_allow_html=True
            )
        else:
            df_fb = pd.DataFrame(fb)
            y_pred = df_fb["prediccion"].tolist()
            y_real = df_fb["real"].tolist()

            # ── Métricas resumen ──────────────────────────────────────────
            accuracy   = sum(p == r for p, r in zip(y_pred, y_real)) / len(y_pred)
            at_risk_tp = sum(1 for p, r in zip(y_pred, y_real) if r == "At Risk" and p == "At Risk")
            at_risk_fn = sum(1 for p, r in zip(y_pred, y_real) if r == "At Risk" and p != "At Risk")
            at_risk_fp = sum(1 for p, r in zip(y_pred, y_real) if r != "At Risk" and p == "At Risk")
            recall_ar  = at_risk_tp / (at_risk_tp + at_risk_fn) if (at_risk_tp + at_risk_fn) > 0 else None
            prec_ar    = at_risk_tp / (at_risk_tp + at_risk_fp) if (at_risk_tp + at_risk_fp) > 0 else None
            n_atrisk   = sum(1 for r in y_real if r == "At Risk")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy global",    f"{accuracy*100:.1f}%")
            c2.metric("Recall At Risk",     f"{recall_ar*100:.1f}%" if recall_ar is not None else "—",
                    delta="≥55% objetivo" if recall_ar is not None and recall_ar >= 0.55 else "bajo objetivo" if recall_ar is not None else None,
                    delta_color="normal" if recall_ar is not None and recall_ar >= 0.55 else "inverse")
            c3.metric("Precisión At Risk",  f"{prec_ar*100:.1f}%"  if prec_ar  is not None else "—")
            c4.metric("Casos At Risk reales", n_atrisk)

            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

            col_left, col_right = st.columns(2, gap="large")

            # ── Matriz de confusión ───────────────────────────────────────
            with col_left:
                st.markdown("**Matriz de confusión**")
                conf_data = {c: {c2: 0 for c2 in CLASSES_EN} for c in CLASSES_EN}
                for p, r in zip(y_pred, y_real):
                    if r in conf_data and p in conf_data:
                        conf_data[r][p] += 1

                rows = []
                for real_cls in CLASSES_EN:
                    row = {"Real \\ Pred": EN_TO_ES[real_cls]}
                    for pred_cls in CLASSES_EN:
                        row[EN_TO_ES[pred_cls]] = conf_data[real_cls][pred_cls]
                    rows.append(row)

                df_conf = pd.DataFrame(rows).set_index("Real \\ Pred")

                # Colorear diagonal
                def highlight_diag(df):
                    styles = pd.DataFrame("", index=df.index, columns=df.columns)
                    for i, col in enumerate(df.columns):
                        if i < len(df.index):
                            styles.iloc[i, i] = "background-color:#14532d;color:#4ade80;font-weight:bold"
                    return styles

                st.dataframe(
                    df_conf.style.apply(highlight_diag, axis=None),
                    use_container_width=True
                )

            # ── Distribución de errores por clase ─────────────────────────
            with col_right:
                st.markdown("**Distribución de errores por clase**")
                error_data = []
                for cls in CLASSES_EN:
                    total  = sum(1 for r in y_real if r == cls)
                    errors = sum(1 for p, r in zip(y_pred, y_real) if r == cls and p != cls)
                    if total > 0:
                        error_data.append({
                            "Clase":       EN_TO_ES[cls],
                            "Total":       total,
                            "Errores":     errors,
                            "Aciertos":    total - errors,
                            "Error (%)":   round(errors / total * 100, 1),
                        })

                if error_data:
                    df_err = pd.DataFrame(error_data)
                    for _, row in df_err.iterrows():
                        pct_ok  = (row["Aciertos"] / row["Total"]) * 100
                        pct_err = row["Error (%)"]
                        color   = "#f87171" if row["Clase"] == "En Riesgo" else "#60a5fa"
                        st.markdown(
                            f"<div style='margin-bottom:0.7rem;'>"
                            f"<div style='display:flex;justify-content:space-between;margin-bottom:0.25rem;'>"
                            f"<span style='color:#e5e7eb;font-size:0.85rem;'>{row['Clase']}</span>"
                            f"<span style='color:#9ca3af;font-size:0.82rem;'>{row['Errores']}/{row['Total']} errores ({pct_err}%)</span>"
                            f"</div>"
                            f"<div style='background:#1e2535;border-radius:4px;height:8px;overflow:hidden;'>"
                            f"<div style='width:{pct_ok:.1f}%;background:{color};height:100%;border-radius:4px;'></div>"
                            f"</div></div>",
                            unsafe_allow_html=True
                        )

            # ── Alerta si Recall At Risk < 0.55 ───────────────────────────
            if recall_ar is not None and recall_ar < 0.55:
                st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
                st.markdown(
                    "<div style='background:#1f0505;border:1px solid #7f1d1d;border-radius:8px;"
                    "padding:0.8rem 1rem;color:#f87171;font-size:0.88rem;'>"
                    "⚠️ <strong>Alerta:</strong> El Recall de At Risk está por debajo del umbral objetivo (55%). "
                    "Considera revisar el umbral de clasificación o reentrenar el modelo con nuevos datos."
                    "</div>",
                    unsafe_allow_html=True
                )

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 5 — INFORMACIÓN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ Información":
    st.markdown('<div class="main-title">Sobre el modelo</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Cómo funciona PawPredict y cómo interpretar los resultados</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(
            "<div class='info-section'>"
            "<div class='info-title'>🧠 ¿Qué hace el modelo?</div>"
            "<div style='color:#9ca3af;font-size:0.9rem;line-height:1.7;'>"
            "El modelo analiza las características de cada animal en el momento de su ingreso "
            "al refugio y predice cuál de los 4 outcomes es más probable:<br><br>"
            "<strong style='color:#4ade80;'>🏠 Adopción</strong> — el animal será adoptado<br>"
            "<strong style='color:#60a5fa;'>🚐 Traslado</strong> — será transferido a otro centro<br>"
            "<strong style='color:#fbbf24;'>🔄 Devolución</strong> — será devuelto a su dueño<br>"
            "<strong style='color:#f87171;'>⚠️ At Risk</strong> — necesita intervención urgente"
            "</div></div>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<div class='info-section'>"
            "<div class='info-title'>📊 Variables que usa el modelo</div>"
            "<div style='color:#9ca3af;font-size:0.9rem;line-height:1.8;'>"
            "• <strong style='color:#e5e7eb;'>Tipo de animal</strong> — Perro o Gato<br>"
            "• <strong style='color:#e5e7eb;'>Sexo</strong> — incluye si está esterilizado<br>"
            "• <strong style='color:#e5e7eb;'>Tipo de ingreso</strong> — cómo llegó al refugio<br>"
            "• <strong style='color:#e5e7eb;'>Condición de ingreso</strong> — su estado al llegar<br>"
            "• <strong style='color:#e5e7eb;'>Grupo de edad</strong> — de cachorro a senior<br>"
            "• <strong style='color:#e5e7eb;'>Tipo de raza</strong> — mestizo o raza pura<br>"
            "• <strong style='color:#e5e7eb;'>Color</strong> — mono, bi o tricolor<br>"
            "• <strong style='color:#e5e7eb;'>Estación</strong> — época del año del ingreso"
            "</div></div>",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            "<div class='info-section'>"
            "<div class='info-title'>⚠️ Niveles de riesgo</div>"
            "<div style='color:#9ca3af;font-size:0.9rem;line-height:1.8;'>"
            "El nivel de riesgo se calcula a partir de la probabilidad que el modelo "
            "asigna a la clase <em>At Risk</em>:<br><br>"
            "<span class='badge-low'>Riesgo Bajo</span> &nbsp; P(At Risk) &lt; 25% — monitoreo estándar<br><br>"
            "<span class='badge-medium'>Riesgo Medio</span> &nbsp; P(At Risk) 25–55% — intervención preventiva<br><br>"
            "<span class='badge-high'>Riesgo Alto</span> &nbsp; P(At Risk) &gt; 55% — acción urgente"
            "</div></div>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<div class='info-section'>"
            "<div class='info-title'>⚙️ Detalles técnicos</div>"
            "<div style='color:#9ca3af;font-size:0.9rem;line-height:1.8;'>"
            "• Entrenado con <strong style='color:#e5e7eb;'>110.000+ registros</strong> del Austin Animal Center<br>"
            "• Clasificación multiclase con <strong style='color:#e5e7eb;'>4 outcomes</strong><br>"
            "• Algoritmo: <strong style='color:#e5e7eb;'>XGBoost + sample_weight</strong><br>"
            "• Optimización: <strong style='color:#e5e7eb;'>Optuna 75 trials</strong><br>"
            "• Balanceo: <strong style='color:#e5e7eb;'>compute_sample_weight('balanced')</strong><br>"
            "• Métrica Optuna: <strong style='color:#e5e7eb;'>70% Recall At Risk + 30% F1-macro</strong><br>"
            "• Umbral At Risk: <strong style='color:#e5e7eb;'>ajustado automáticamente</strong><br>"
            "• Preprocesado: <strong style='color:#e5e7eb;'>RobustScaler + OHE + OrdinalEncoder</strong>"
            "</div></div>",
            unsafe_allow_html=True
        )

    st.markdown(
        "<div class='info-section' style='border-color:#7f1d1d;'>"
        "<div class='info-title' style='color:#fca5a5;'>⚠️ Limitaciones importantes</div>"
        "<div style='color:#9ca3af;font-size:0.9rem;line-height:1.7;'>"
        "Este modelo es una herramienta de <strong style='color:#e5e7eb;'>apoyo a la decisión</strong>, "
        "no un sustituto del criterio veterinario o del personal del refugio. "
        "Las predicciones se basan en patrones históricos y pueden no reflejar la situación individual de cada animal. "
        "Úsalas como punto de partida para priorizar recursos, no como sentencia definitiva."
        "</div></div>",
        unsafe_allow_html=True
    )
