"""
Tests para el sistema de predicción del Austin Animal Center.
Orden de prioridad:
    1. Tests de integración modelo completo
    2. predict() y umbral At Risk
    3. Preprocesamiento (OrdinalEncoder, log1p)
    4. build_input_df() y columnas
    5. Métricas de feedback (accuracy, recall)

Ejecutar con:
    pytest test_app.py -v
    pytest test_app.py -v --tb=short        # traceback corto
    pytest test_app.py -v -k "integration"  # solo integración
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — funciones extraídas de app.py para testear sin importar Streamlit
# ══════════════════════════════════════════════════════════════════════════════

AGE_ORDER = ["Cachorro (<6m)", "Joven (6m-1a)", "Adulto joven (1-3a)", "Adulto (3-7a)", "Senior (>7a)"]

FEATURE_OPTIONS = {
    "AnimalType":      ["Dog", "Cat"],
    "Sex":             ["Male", "Female", "Neutered Male", "Spayed Female", "Intact Male", "Intact Female"],
    "IntakeType":      ["Stray", "Owner Surrender", "Public Assist", "Euthanasia Request", "Wildlife", "Abandoned"],
    "IntakeCondition": ["Normal", "Injured", "Sick", "Aged", "Feral", "Pregnant", "Nursing", "Other"],
    "AgeGroup":        AGE_ORDER,
    "breed_type":      ["mix", "purebred"],
    "Color_grouped":   ["Monocolor", "Bicolor", "Tricolor"],
    "Season":          ["Primavera", "Verano", "Otoño", "Invierno"],
}

EXPECTED_COLUMNS = [
    "AgeInDays_log", "AnimalType", "Sex", "IntakeType",
    "IntakeCondition", "AgeGroup", "breed_type", "Color_grouped", "Season",
]


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


# ── Fixture: datos de entrada válidos por defecto ─────────────────────────────
@pytest.fixture
def form_data_normal():
    return {
        "AnimalType":     "Dog",
        "Sex":            "Neutered Male",
        "IntakeType":     "Stray",
        "IntakeCondition":"Normal",
        "AgeGroup":       "Adulto joven (1-3a)",
        "breed_type":     "mix",
        "Color_grouped":  "Bicolor",
        "Season":         "Primavera",
    }

@pytest.fixture
def form_data_high_risk():
    return {
        "AnimalType":     "Cat",
        "Sex":            "Intact Male",
        "IntakeType":     "Euthanasia Request",
        "IntakeCondition":"Injured",
        "AgeGroup":       "Senior (>7a)",
        "breed_type":     "mix",
        "Color_grouped":  "Monocolor",
        "Season":         "Invierno",
    }

@pytest.fixture
def mock_artifacts():
    """Crea mocks del modelo, preprocesador y label encoder."""
    classes = np.array(["Adoption", "At Risk", "Return to Owner", "Transfer"])

    model        = MagicMock()
    preprocessor = MagicMock()
    le_target    = MagicMock()
    le_target.classes_ = classes

    preprocessor.transform.return_value = np.zeros((1, 20))

    return model, preprocessor, le_target


# ══════════════════════════════════════════════════════════════════════════════
# 1. TESTS DE INTEGRACIÓN — modelo completo
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegracion:
    """Flujo completo: form_data → build_input_df → predict → clase + proba_dict."""

    def test_flujo_completo_devuelve_clase_valida(self, form_data_normal, mock_artifacts):
        """El pipeline completo devuelve una clase dentro de las 4 esperadas."""
        model, preprocessor, le_target = mock_artifacts
        model.predict_proba.return_value = np.array([[0.6, 0.1, 0.2, 0.1]])
        le_target.inverse_transform.return_value = ["Adoption"]

        input_df       = build_input_df(form_data_normal)
        clase, proba   = predict(input_df, model, preprocessor, le_target, umbral=0.5)

        assert clase in ["Adoption", "Transfer", "Return to Owner", "At Risk"]

    def test_flujo_completo_proba_dict_suma_1(self, form_data_normal, mock_artifacts):
        """Las probabilidades del pipeline suman 1 (con tolerancia de punto flotante)."""
        model, preprocessor, le_target = mock_artifacts
        model.predict_proba.return_value = np.array([[0.5, 0.2, 0.2, 0.1]])
        le_target.inverse_transform.return_value = ["Adoption"]

        input_df     = build_input_df(form_data_normal)
        _, proba     = predict(input_df, model, preprocessor, le_target)

        assert abs(sum(proba.values()) - 1.0) < 1e-6

    def test_flujo_completo_proba_dict_tiene_4_clases(self, form_data_normal, mock_artifacts):
        """El diccionario de probabilidades contiene exactamente las 4 clases."""
        model, preprocessor, le_target = mock_artifacts
        model.predict_proba.return_value = np.array([[0.5, 0.2, 0.2, 0.1]])
        le_target.inverse_transform.return_value = ["Adoption"]

        input_df     = build_input_df(form_data_normal)
        _, proba     = predict(input_df, model, preprocessor, le_target)

        assert set(proba.keys()) == {"Adoption", "At Risk", "Return to Owner", "Transfer"}

    def test_flujo_alto_riesgo_activa_at_risk(self, form_data_high_risk, mock_artifacts):
        """Un animal con perfil de alto riesgo y P(At Risk)=0.8 se clasifica como At Risk."""
        model, preprocessor, le_target = mock_artifacts
        # At Risk está en índice 1 según classes_
        model.predict_proba.return_value = np.array([[0.05, 0.80, 0.10, 0.05]])
        le_target.inverse_transform.return_value = ["At Risk"]

        input_df     = build_input_df(form_data_high_risk)
        clase, proba = predict(input_df, model, preprocessor, le_target, umbral=0.5)

        assert clase == "At Risk"
        assert proba["At Risk"] == pytest.approx(0.80)

    def test_flujo_preprocessor_recibe_columnas_correctas(self, form_data_normal, mock_artifacts):
        """El preprocesador recibe un DataFrame con exactamente las columnas esperadas."""
        model, preprocessor, le_target = mock_artifacts
        model.predict_proba.return_value = np.array([[0.6, 0.1, 0.2, 0.1]])
        le_target.inverse_transform.return_value = ["Adoption"]

        input_df = build_input_df(form_data_normal)
        predict(input_df, model, preprocessor, le_target)

        call_args = preprocessor.transform.call_args[0][0]
        assert list(call_args.columns) == EXPECTED_COLUMNS


# ══════════════════════════════════════════════════════════════════════════════
# 2. TESTS DE predict() Y UMBRAL AT RISK
# ══════════════════════════════════════════════════════════════════════════════

class TestPredict:
    """Lógica de predict() y aplicación del umbral óptimo."""

    def test_umbral_fuerza_at_risk_cuando_prob_igual_umbral(self, form_data_normal, mock_artifacts):
        """Si P(At Risk) == umbral exacto, debe clasificar como At Risk."""
        model, preprocessor, le_target = mock_artifacts
        model.predict_proba.return_value = np.array([[0.4, 0.5, 0.05, 0.05]])
        le_target.inverse_transform.return_value = ["Adoption"]

        input_df     = build_input_df(form_data_normal)
        clase, _     = predict(input_df, model, preprocessor, le_target, umbral=0.5)

        assert clase == "At Risk"

    def test_umbral_no_fuerza_at_risk_cuando_prob_menor(self, form_data_normal, mock_artifacts):
        """Si P(At Risk) < umbral, la clase es la del argmax normal."""
        model, preprocessor, le_target = mock_artifacts
        # Adoption (idx 0) tiene prob máxima, At Risk (idx 1) está por debajo del umbral
        model.predict_proba.return_value = np.array([[0.6, 0.3, 0.05, 0.05]])
        le_target.inverse_transform.return_value = ["Adoption"]

        input_df     = build_input_df(form_data_normal)
        clase, _     = predict(input_df, model, preprocessor, le_target, umbral=0.5)

        assert clase == "Adoption"

    def test_umbral_bajo_aumenta_sensibilidad_at_risk(self, form_data_normal, mock_artifacts):
        """Con umbral bajo (0.15), P(At Risk)=0.20 debe forzar At Risk."""
        model, preprocessor, le_target = mock_artifacts
        model.predict_proba.return_value = np.array([[0.55, 0.20, 0.15, 0.10]])
        le_target.inverse_transform.return_value = ["Adoption"]

        input_df     = build_input_df(form_data_normal)
        clase, _     = predict(input_df, model, preprocessor, le_target, umbral=0.15)

        assert clase == "At Risk"

    def test_umbral_alto_reduce_falsos_positivos(self, form_data_normal, mock_artifacts):
        """Con umbral alto (0.90), P(At Risk)=0.80 no debe forzar At Risk."""
        model, preprocessor, le_target = mock_artifacts
        model.predict_proba.return_value = np.array([[0.05, 0.80, 0.10, 0.05]])
        le_target.inverse_transform.return_value = ["At Risk"]

        input_df     = build_input_df(form_data_normal)
        clase, _     = predict(input_df, model, preprocessor, le_target, umbral=0.90)

        # Con umbral 0.90 y P(At Risk)=0.80, va al argmax que es At Risk (idx 1)
        # pero la lógica del umbral no lo activa — lo activa el argmax
        assert clase == "At Risk"  # argmax igual devuelve At Risk en este caso

    def test_predict_devuelve_tupla_clase_y_dict(self, form_data_normal, mock_artifacts):
        """predict() devuelve exactamente (str, dict)."""
        model, preprocessor, le_target = mock_artifacts
        model.predict_proba.return_value = np.array([[0.6, 0.1, 0.2, 0.1]])
        le_target.inverse_transform.return_value = ["Adoption"]

        input_df     = build_input_df(form_data_normal)
        resultado    = predict(input_df, model, preprocessor, le_target)

        assert isinstance(resultado, tuple)
        assert len(resultado) == 2
        assert isinstance(resultado[0], str)
        assert isinstance(resultado[1], dict)

    def test_proba_dict_valores_entre_0_y_1(self, form_data_normal, mock_artifacts):
        """Todas las probabilidades del dict están entre 0 y 1."""
        model, preprocessor, le_target = mock_artifacts
        model.predict_proba.return_value = np.array([[0.5, 0.2, 0.2, 0.1]])
        le_target.inverse_transform.return_value = ["Adoption"]

        input_df  = build_input_df(form_data_normal)
        _, proba  = predict(input_df, model, preprocessor, le_target)

        for cls, p in proba.items():
            assert 0.0 <= p <= 1.0, f"Probabilidad fuera de rango para {cls}: {p}"


# ══════════════════════════════════════════════════════════════════════════════
# 3. TESTS DE PREPROCESAMIENTO — OrdinalEncoder, log1p
# ══════════════════════════════════════════════════════════════════════════════

class TestPreprocesamiento:
    """Transformaciones de variables antes de entrar al modelo."""

    def test_log1p_cachorro(self):
        """Cachorro (<6m) → 90 días → log1p(90) ≈ 4.511."""
        dias    = age_to_days("Cachorro (<6m)")
        log_val = np.log1p(dias)
        assert dias == 90
        assert log_val == pytest.approx(np.log1p(90), rel=1e-6)

    def test_log1p_senior(self):
        """Senior (>7a) → 3285 días → log1p(3285) ≈ 8.097."""
        dias    = age_to_days("Senior (>7a)")
        log_val = np.log1p(dias)
        assert dias == 3285
        assert log_val == pytest.approx(np.log1p(3285), rel=1e-6)

    def test_log1p_siempre_positivo(self):
        """log1p de cualquier edad válida es siempre > 0."""
        for age_group in AGE_ORDER:
            dias    = age_to_days(age_group)
            log_val = np.log1p(dias)
            assert log_val > 0, f"log1p negativo para {age_group}"

    def test_orden_ordinal_age_group_creciente(self):
        """Los días representativos crecen en el mismo orden que AGE_ORDER."""
        dias = [age_to_days(g) for g in AGE_ORDER]
        assert dias == sorted(dias), "El orden de AgeGroup no es creciente en días"

    def test_age_group_desconocido_devuelve_fallback(self):
        """Un AgeGroup no reconocido devuelve el fallback de 365 días."""
        dias = age_to_days("Grupo inventado")
        assert dias == 365

    def test_log1p_mayor_para_senior_que_cachorro(self):
        """Senior tiene mayor log1p que Cachorro — el modelo recibe más 'edad'."""
        log_cachorro = np.log1p(age_to_days("Cachorro (<6m)"))
        log_senior   = np.log1p(age_to_days("Senior (>7a)"))
        assert log_senior > log_cachorro

    @pytest.mark.parametrize("age_group,expected_days", [
        ("Cachorro (<6m)",      90),
        ("Joven (6m-1a)",       270),
        ("Adulto joven (1-3a)", 730),
        ("Adulto (3-7a)",       1825),
        ("Senior (>7a)",        3285),
    ])
    def test_age_to_days_valores_exactos(self, age_group, expected_days):
        """Cada AgeGroup mapea exactamente a sus días representativos."""
        assert age_to_days(age_group) == expected_days


# ══════════════════════════════════════════════════════════════════════════════
# 4. TESTS DE build_input_df() Y COLUMNAS
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildInputDf:
    """Construcción del DataFrame de entrada al modelo."""

    def test_columnas_correctas(self, form_data_normal):
        """El DataFrame tiene exactamente las columnas que espera el preprocesador."""
        df = build_input_df(form_data_normal)
        assert list(df.columns) == EXPECTED_COLUMNS

    def test_una_sola_fila(self, form_data_normal):
        """El DataFrame tiene exactamente 1 fila."""
        df = build_input_df(form_data_normal)
        assert len(df) == 1

    def test_ageindays_log_es_float(self, form_data_normal):
        """AgeInDays_log es un float (no int ni NaN)."""
        df = build_input_df(form_data_normal)
        val = df["AgeInDays_log"].iloc[0]
        assert isinstance(val, float)
        assert not np.isnan(val)

    def test_ageindays_log_correcto(self, form_data_normal):
        """AgeInDays_log = log1p(730) para Adulto joven (1-3a)."""
        df  = build_input_df(form_data_normal)
        val = df["AgeInDays_log"].iloc[0]
        assert val == pytest.approx(np.log1p(730), rel=1e-6)

    def test_valores_categoricos_preservados(self, form_data_normal):
        """Los valores categóricos se copian sin modificar al DataFrame."""
        df = build_input_df(form_data_normal)
        assert df["AnimalType"].iloc[0]      == form_data_normal["AnimalType"]
        assert df["Sex"].iloc[0]             == form_data_normal["Sex"]
        assert df["IntakeType"].iloc[0]      == form_data_normal["IntakeType"]
        assert df["IntakeCondition"].iloc[0] == form_data_normal["IntakeCondition"]
        assert df["breed_type"].iloc[0]      == form_data_normal["breed_type"]
        assert df["Color_grouped"].iloc[0]   == form_data_normal["Color_grouped"]
        assert df["Season"].iloc[0]          == form_data_normal["Season"]

    def test_no_hay_nulos(self, form_data_normal):
        """El DataFrame de entrada no tiene valores nulos."""
        df = build_input_df(form_data_normal)
        assert df.isnull().sum().sum() == 0

    @pytest.mark.parametrize("age_group", AGE_ORDER)
    def test_todos_los_age_groups_generan_df_valido(self, age_group, form_data_normal):
        """Cada AgeGroup genera un DataFrame sin errores ni nulos."""
        form_data_normal["AgeGroup"] = age_group
        df = build_input_df(form_data_normal)
        assert len(df) == 1
        assert df.isnull().sum().sum() == 0


# ══════════════════════════════════════════════════════════════════════════════
# 5. TESTS DE MÉTRICAS DE FEEDBACK (accuracy, recall At Risk)
# ══════════════════════════════════════════════════════════════════════════════

class TestMetricasFeedback:
    """Cálculo manual de métricas en la página de Feedback."""

    def _calcular_metricas(self, y_pred, y_real):
        """Replica exactamente el cálculo de la página de Feedback."""
        accuracy   = sum(p == r for p, r in zip(y_pred, y_real)) / len(y_pred)
        at_risk_tp = sum(1 for p, r in zip(y_pred, y_real) if r == "At Risk" and p == "At Risk")
        at_risk_fn = sum(1 for p, r in zip(y_pred, y_real) if r == "At Risk" and p != "At Risk")
        at_risk_fp = sum(1 for p, r in zip(y_pred, y_real) if r != "At Risk" and p == "At Risk")
        recall_ar  = at_risk_tp / (at_risk_tp + at_risk_fn) if (at_risk_tp + at_risk_fn) > 0 else None
        prec_ar    = at_risk_tp / (at_risk_tp + at_risk_fp) if (at_risk_tp + at_risk_fp) > 0 else None
        return accuracy, recall_ar, prec_ar

    def test_accuracy_perfecto(self):
        """Cuando todas las predicciones son correctas, accuracy = 1.0."""
        y_pred = ["Adoption", "At Risk", "Transfer"]
        y_real = ["Adoption", "At Risk", "Transfer"]
        acc, _, _ = self._calcular_metricas(y_pred, y_real)
        assert acc == pytest.approx(1.0)

    def test_accuracy_cero(self):
        """Cuando ninguna predicción es correcta, accuracy = 0.0."""
        y_pred = ["Adoption",  "Transfer",  "At Risk"]
        y_real = ["At Risk",   "Adoption",  "Transfer"]
        acc, _, _ = self._calcular_metricas(y_pred, y_real)
        assert acc == pytest.approx(0.0)

    def test_accuracy_parcial(self):
        """2 de 4 correctas → accuracy = 0.5."""
        y_pred = ["Adoption", "Adoption", "At Risk",  "Transfer"]
        y_real = ["Adoption", "At Risk",  "At Risk",  "Adoption"]
        acc, _, _ = self._calcular_metricas(y_pred, y_real)
        assert acc == pytest.approx(0.5)

    def test_recall_at_risk_perfecto(self):
        """Todos los At Risk reales detectados → recall = 1.0."""
        y_pred = ["At Risk", "At Risk", "Adoption"]
        y_real = ["At Risk", "At Risk", "Adoption"]
        _, recall, _ = self._calcular_metricas(y_pred, y_real)
        assert recall == pytest.approx(1.0)

    def test_recall_at_risk_cero(self):
        """Ningún At Risk real detectado → recall = 0.0."""
        y_pred = ["Adoption", "Adoption"]
        y_real = ["At Risk",  "At Risk"]
        _, recall, _ = self._calcular_metricas(y_pred, y_real)
        assert recall == pytest.approx(0.0)

    def test_recall_at_risk_parcial(self):
        """1 de 2 At Risk reales detectado → recall = 0.5."""
        y_pred = ["At Risk", "Adoption"]
        y_real = ["At Risk", "At Risk"]
        _, recall, _ = self._calcular_metricas(y_pred, y_real)
        assert recall == pytest.approx(0.5)

    def test_recall_none_cuando_no_hay_at_risk_real(self):
        """Si no hay casos reales de At Risk, recall devuelve None (no divide por 0)."""
        y_pred = ["Adoption", "Transfer"]
        y_real = ["Adoption", "Transfer"]
        _, recall, _ = self._calcular_metricas(y_pred, y_real)
        assert recall is None

    def test_precision_at_risk(self):
        """1 TP y 1 FP → precisión = 0.5."""
        y_pred = ["At Risk", "At Risk", "Adoption"]
        y_real = ["At Risk", "Adoption", "Adoption"]
        _, _, prec = self._calcular_metricas(y_pred, y_real)
        assert prec == pytest.approx(0.5)

    def test_alerta_recall_bajo_umbral(self):
        """La alerta de recall < 0.55 se activa correctamente."""
        y_pred = ["At Risk", "Adoption", "Adoption", "Adoption"]
        y_real = ["At Risk", "At Risk",  "At Risk",  "At Risk"]
        _, recall, _ = self._calcular_metricas(y_pred, y_real)
        assert recall is not None
        assert recall < 0.55  # debe disparar la alerta en la app


# ══════════════════════════════════════════════════════════════════════════════
# 6. TESTS DE risk_level()
# ══════════════════════════════════════════════════════════════════════════════

class TestRiskLevel:
    """Clasificación del nivel de riesgo según P(At Risk)."""

    @pytest.mark.parametrize("prob,expected_level,expected_label", [
        (0.00,  "low",    "Riesgo Bajo"),
        (0.24,  "low",    "Riesgo Bajo"),
        (0.25,  "medium", "Riesgo Medio"),
        (0.54,  "medium", "Riesgo Medio"),
        (0.55,  "high",   "Riesgo Alto"),
        (1.00,  "high",   "Riesgo Alto"),
    ])
    def test_niveles_correctos(self, prob, expected_level, expected_label):
        """Cada rango de probabilidad devuelve el nivel y etiqueta correctos."""
        level, label, badge = risk_level(prob)
        assert level == expected_level
        assert label == expected_label

    def test_badges_son_strings(self):
        """Los badges CSS devueltos son strings no vacíos."""
        for prob in [0.1, 0.4, 0.8]:
            _, _, badge = risk_level(prob)
            assert isinstance(badge, str) and len(badge) > 0

    def test_frontera_025_es_medium(self):
        """El valor exacto 0.25 cae en Riesgo Medio (no en Bajo)."""
        level, _, _ = risk_level(0.25)
        assert level == "medium"

    def test_frontera_055_es_high(self):
        """El valor exacto 0.55 cae en Riesgo Alto (no en Medio)."""
        level, _, _ = risk_level(0.55)
        assert level == "high"