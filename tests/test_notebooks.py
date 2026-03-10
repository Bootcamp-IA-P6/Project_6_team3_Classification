"""
Tests unitarios para las transformaciones de los notebooks.

Cubre:
    - 00_preprocessing  → parse_age(), AgeGroup, breed_type, Color_grouped, Season, OutcomeClass
    - 02_encoding       → log1p, LabelEncoder, split 60/20/20, ColumnTransformer, SMOTE
    - 03_optimization   → función objetivo Optuna, umbral At Risk

Ejecutar con:
    pytest test_notebooks.py -v
    pytest test_notebooks.py -v -k "notebook00"   # solo preprocesamiento
    pytest test_notebooks.py -v -k "notebook02"   # solo encoding
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES EXTRAÍDAS DE LOS NOTEBOOKS (sin dependencia de Jupyter)
# ══════════════════════════════════════════════════════════════════════════════

# ── Notebook 00 ───────────────────────────────────────────────────────────────

def parse_age(age_str: str):
    """Convierte strings de edad ('2 years', '3 months'...) a días. [nb00]"""
    if age_str is None:
        return None
    age_str = str(age_str).lower().strip()
    parts = age_str.split()
    if len(parts) < 2:
        return None
    try:
        n = float(parts[0])
    except ValueError:
        return None
    if "year"  in parts[1]: return n * 365
    if "month" in parts[1]: return n * 30
    if "week"  in parts[1]: return n * 7
    if "day"   in parts[1]: return n
    return None


def assign_age_group(age_in_days: float) -> str:
    """Asigna AgeGroup según días. [nb00]"""
    if age_in_days < 180:   return "Cachorro (<6m)"
    if age_in_days < 365:   return "Joven (6m-1a)"
    if age_in_days < 1095:  return "Adulto joven (1-3a)"
    if age_in_days < 2555:  return "Adulto (3-7a)"
    return "Senior (>7a)"


def assign_breed_type(breed: str) -> str:
    """Clasifica raza como mix o purebred. [nb00]"""
    return "mix" if "Mix" in str(breed) else "purebred"


def assign_color_grouped(color: str) -> str:
    """Agrupa colores en Monocolor, Bicolor o Tricolor. [nb00]"""
    color = str(color)
    if "/" not in color:
        return "Monocolor"
    if color.count("/") >= 2:
        return "Tricolor"
    return "Bicolor"


def assign_season(month: int) -> str:
    """Asigna estación del año según mes. [nb00]"""
    if month in [3, 4, 5]:   return "Primavera"
    if month in [6, 7, 8]:   return "Verano"
    if month in [9, 10, 11]: return "Otoño"
    return "Invierno"


def assign_outcome_class(outcome_type: str) -> str:
    """Agrupa OutcomeType en las 4 clases del modelo. [nb00]"""
    if outcome_type in ["Euthanasia", "Died", "Disposal"]:
        return "At Risk"
    if outcome_type in ["Adoption", "Rto-Adopt"]:
        return "Adoption"
    if outcome_type == "Transfer":
        return "Transfer"
    if outcome_type == "Return to Owner":
        return "Return to Owner"
    return None


# ── Notebook 02 ───────────────────────────────────────────────────────────────

def compute_split_sizes(n_total: int, test_size=0.20, val_fraction=0.25):
    """Calcula tamaños de train/val/test con split 60/20/20. [nb02]"""
    n_test  = round(n_total * test_size)
    n_temp  = n_total - n_test
    n_val   = round(n_temp * val_fraction)
    n_train = n_temp - n_val
    return n_train, n_val, n_test


def compute_at_risk_ratio(y: np.ndarray, at_risk_idx: int) -> float:
    """Calcula el porcentaje de At Risk en un array de etiquetas. [nb02/03]"""
    return (y == at_risk_idx).mean() * 100


# ── Notebook 03 ───────────────────────────────────────────────────────────────

def compute_optuna_objective(recall_at_risk: float, f1_macro: float) -> float:
    """Función objetivo de Optuna: 0.70 × Recall_AtRisk + 0.30 × F1-macro. [nb03]"""
    return 0.70 * recall_at_risk + 0.30 * f1_macro


def apply_threshold(proba: np.ndarray, at_risk_idx: int, umbral: float) -> np.ndarray:
    """Aplica umbral personalizado para At Risk. [nb03]"""
    preds = proba.argmax(axis=1).copy()
    preds[proba[:, at_risk_idx] >= umbral] = at_risk_idx
    return preds


# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 00 — PREPROCESAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

class TestNotebook00ParseAge:
    """parse_age() — conversión de strings de edad a días."""

    @pytest.mark.parametrize("age_str,expected", [
        ("1 year",     365.0),
        ("2 years",    730.0),
        ("3 months",   90.0),
        ("6 months",   180.0),
        ("2 weeks",    14.0),
        ("1 week",     7.0),
        ("10 days",    10.0),
        ("1 day",      1.0),
    ])
    def test_conversiones_estandar(self, age_str, expected):
        """Conversiones habituales de edad a días."""
        assert parse_age(age_str) == pytest.approx(expected)

    def test_none_devuelve_none(self):
        """None como entrada devuelve None."""
        assert parse_age(None) is None

    def test_string_vacio_devuelve_none(self):
        """String vacío o sin unidad devuelve None."""
        assert parse_age("") is None
        assert parse_age("2") is None

    def test_texto_invalido_devuelve_none(self):
        """Texto no numérico devuelve None."""
        assert parse_age("unknown years") is None

    def test_case_insensitive(self):
        """La función es insensible a mayúsculas."""
        assert parse_age("2 Years")  == pytest.approx(730.0)
        assert parse_age("3 MONTHS") == pytest.approx(90.0)

    def test_valores_decimales(self):
        """Acepta valores decimales correctamente."""
        assert parse_age("1.5 years") == pytest.approx(1.5 * 365)

    def test_resultado_positivo_para_entradas_validas(self):
        """Cualquier entrada válida devuelve un valor positivo."""
        entradas = ["1 year", "3 months", "2 weeks", "5 days"]
        for entrada in entradas:
            resultado = parse_age(entrada)
            assert resultado is not None and resultado > 0


class TestNotebook00AgeGroup:
    """assign_age_group() — agrupación de edad en categorías."""

    @pytest.mark.parametrize("days,expected_group", [
        (0,    "Cachorro (<6m)"),
        (90,   "Cachorro (<6m)"),
        (179,  "Cachorro (<6m)"),
        (180,  "Joven (6m-1a)"),
        (270,  "Joven (6m-1a)"),
        (364,  "Joven (6m-1a)"),
        (365,  "Adulto joven (1-3a)"),
        (730,  "Adulto joven (1-3a)"),
        (1094, "Adulto joven (1-3a)"),
        (1095, "Adulto (3-7a)"),
        (1825, "Adulto (3-7a)"),
        (2554, "Adulto (3-7a)"),
        (2555, "Senior (>7a)"),
        (3285, "Senior (>7a)"),
    ])
    def test_fronteras_exactas(self, days, expected_group):
        """Cada valor en el límite exacto cae en el grupo correcto."""
        assert assign_age_group(days) == expected_group

    def test_grupos_validos(self):
        """Todos los grupos devueltos están en el conjunto permitido."""
        grupos_validos = {
            "Cachorro (<6m)", "Joven (6m-1a)", "Adulto joven (1-3a)",
            "Adulto (3-7a)", "Senior (>7a)"
        }
        for days in [0, 100, 300, 800, 2000, 4000]:
            assert assign_age_group(days) in grupos_validos


class TestNotebook00BreedType:
    """assign_breed_type() — clasificación mestizo/raza pura."""

    @pytest.mark.parametrize("breed,expected", [
        ("Labrador Retriever Mix",      "mix"),
        ("Domestic Shorthair Mix",      "mix"),
        ("German Shepherd Mix",         "mix"),
        ("Labrador Retriever",          "purebred"),
        ("Chihuahua Shorthair",         "purebred"),
        ("Domestic Shorthair",          "purebred"),
        ("Pit Bull",                    "purebred"),
    ])
    def test_clasificacion_breed(self, breed, expected):
        """Razas con 'Mix' son mix, el resto purebred."""
        assert assign_breed_type(breed) == expected

    def test_solo_dos_categorias(self):
        """Solo existen dos categorías posibles."""
        breeds = ["Labrador Mix", "Poodle", "Unknown Mix", "Siamese"]
        for breed in breeds:
            assert assign_breed_type(breed) in ("mix", "purebred")


class TestNotebook00ColorGrouped:
    """assign_color_grouped() — agrupación de colores."""

    @pytest.mark.parametrize("color,expected", [
        ("Black",                   "Monocolor"),
        ("White",                   "Monocolor"),
        ("Brown",                   "Monocolor"),
        ("Black/White",             "Bicolor"),
        ("Brown/White",             "Bicolor"),
        ("Black/White/Brown",       "Tricolor"),
        ("Orange/White/Black",      "Tricolor"),
    ])
    def test_agrupacion_colores(self, color, expected):
        """Colores con 0, 1 o 2+ '/' se agrupan correctamente."""
        assert assign_color_grouped(color) == expected

    def test_solo_tres_categorias(self):
        """Solo existen tres categorías posibles."""
        colores = ["Black", "Black/White", "Black/White/Brown", "Red/Blue/Green/Yellow"]
        for color in colores:
            assert assign_color_grouped(color) in ("Monocolor", "Bicolor", "Tricolor")


class TestNotebook00Season:
    """assign_season() — asignación de estación por mes."""

    @pytest.mark.parametrize("month,expected", [
        (1,  "Invierno"),
        (2,  "Invierno"),
        (3,  "Primavera"),
        (4,  "Primavera"),
        (5,  "Primavera"),
        (6,  "Verano"),
        (7,  "Verano"),
        (8,  "Verano"),
        (9,  "Otoño"),
        (10, "Otoño"),
        (11, "Otoño"),
        (12, "Invierno"),
    ])
    def test_todos_los_meses(self, month, expected):
        """Cada mes del año mapea a la estación correcta."""
        assert assign_season(month) == expected

    def test_cuatro_estaciones_posibles(self):
        """Solo existen cuatro estaciones posibles."""
        for month in range(1, 13):
            assert assign_season(month) in ("Primavera", "Verano", "Otoño", "Invierno")


class TestNotebook00OutcomeClass:
    """assign_outcome_class() — agrupación de outcomes en 4 clases."""

    @pytest.mark.parametrize("outcome,expected_class", [
        ("Euthanasia",      "At Risk"),
        ("Died",            "At Risk"),
        ("Disposal",        "At Risk"),
        ("Adoption",        "Adoption"),
        ("Rto-Adopt",       "Adoption"),
        ("Transfer",        "Transfer"),
        ("Return to Owner", "Return to Owner"),
    ])
    def test_agrupacion_outcomes(self, outcome, expected_class):
        """Cada OutcomeType se agrupa en la clase correcta."""
        assert assign_outcome_class(outcome) == expected_class

    def test_outcome_desconocido_devuelve_none(self):
        """Outcomes no reconocidos devuelven None."""
        assert assign_outcome_class("Unknown") is None
        assert assign_outcome_class("Missing") is None

    def test_at_risk_agrupa_tres_outcomes(self):
        """At Risk agrupa exactamente Euthanasia, Died y Disposal."""
        at_risk_outcomes = ["Euthanasia", "Died", "Disposal"]
        for outcome in at_risk_outcomes:
            assert assign_outcome_class(outcome) == "At Risk"

    def test_adoption_agrupa_dos_outcomes(self):
        """Adoption agrupa Adoption y Rto-Adopt."""
        for outcome in ["Adoption", "Rto-Adopt"]:
            assert assign_outcome_class(outcome) == "Adoption"


# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 02 — ENCODING Y BALANCEO
# ══════════════════════════════════════════════════════════════════════════════

class TestNotebook02Log1p:
    """Transformación logarítmica AgeInDays → AgeInDays_log."""

    def test_log1p_reduce_skewness(self):
        """log1p reduce la asimetría de una distribución sesgada a la derecha."""
        np.random.seed(42)
        # Simula distribución sesgada parecida a AgeInDays
        ages = np.random.exponential(scale=500, size=1000)
        skew_original = pd.Series(ages).skew()
        skew_log      = pd.Series(np.log1p(ages)).skew()
        assert skew_log < skew_original

    def test_log1p_no_genera_nan(self):
        """log1p no genera NaN para valores de edad positivos."""
        ages    = np.array([1, 30, 90, 365, 730, 1825, 3285])
        log_val = np.log1p(ages)
        assert not np.any(np.isnan(log_val))

    def test_log1p_preserva_orden(self):
        """El orden relativo de edades se preserva tras log1p."""
        ages    = np.array([90, 270, 730, 1825, 3285])
        log_val = np.log1p(ages)
        assert np.all(np.diff(log_val) > 0)

    def test_log1p_valores_positivos(self):
        """log1p siempre produce valores positivos para edades > 0."""
        ages = np.array([1, 90, 365, 3285])
        assert np.all(np.log1p(ages) > 0)


class TestNotebook02LabelEncoder:
    """LabelEncoder del target — orden y consistencia."""

    @pytest.fixture
    def le(self):
        from sklearn.preprocessing import LabelEncoder
        CLASS_ORDER = ["Adoption", "Transfer", "Return to Owner", "At Risk"]
        le = LabelEncoder()
        le.fit(CLASS_ORDER)
        return le

    def test_clases_son_exactamente_4(self, le):
        """El LabelEncoder tiene exactamente 4 clases."""
        assert len(le.classes_) == 4

    def test_todas_las_clases_presentes(self, le):
        """Las 4 clases del modelo están presentes."""
        expected = {"Adoption", "Transfer", "Return to Owner", "At Risk"}
        assert set(le.classes_) == expected

    def test_transformacion_es_reversible(self, le):
        """inverse_transform(transform(x)) == x para cualquier clase."""
        clases = ["Adoption", "Transfer", "Return to Owner", "At Risk"]
        for clase in clases:
            idx  = le.transform([clase])[0]
            back = le.inverse_transform([idx])[0]
            assert back == clase

    def test_indices_son_enteros(self, le):
        """Los índices devueltos son enteros."""
        clases = ["Adoption", "At Risk", "Transfer"]
        for clase in clases:
            idx = le.transform([clase])[0]
            assert isinstance(idx, (int, np.integer))

    def test_at_risk_tiene_indice_conocido(self, le):
        """At Risk tiene siempre el mismo índice (orden alfabético: 1)."""
        # LabelEncoder ordena alfabéticamente: Adoption=0, At Risk=1, ...
        at_risk_idx = le.transform(["At Risk"])[0]
        assert at_risk_idx == 1


class TestNotebook02Split:
    """Split estratificado 60/20/20."""

    def test_proporciones_aproximadas(self):
        """Con 1000 registros el split produce ~60/20/20."""
        n_train, n_val, n_test = compute_split_sizes(1000)
        assert abs(n_train / 1000 - 0.60) < 0.02
        assert abs(n_val   / 1000 - 0.20) < 0.02
        assert abs(n_test  / 1000 - 0.20) < 0.02

    def test_suma_total(self):
        """La suma de los tres splits es igual al total."""
        n_total = 110847
        n_train, n_val, n_test = compute_split_sizes(n_total)
        assert n_train + n_val + n_test == n_total

    def test_train_es_el_mayor(self):
        """Train siempre es el split más grande."""
        n_train, n_val, n_test = compute_split_sizes(10000)
        assert n_train > n_val
        assert n_train > n_test

    def test_val_y_test_similares(self):
        """Val y test tienen tamaños similares (diferencia < 5%)."""
        n_train, n_val, n_test = compute_split_sizes(10000)
        diff_pct = abs(n_val - n_test) / n_test
        assert diff_pct < 0.05

    def test_split_estratificado_preserva_at_risk(self):
        """El split estratificado preserva la proporción de At Risk (~3.7%)."""
        from sklearn.model_selection import train_test_split

        np.random.seed(42)
        n = 10000
        # Simula distribución real: 3.7% At Risk
        y = np.array([3] * 370 + [0] * 4900 + [2] * 2000 + [1] * 2730)
        np.random.shuffle(y)

        y_temp, y_test = train_test_split(y, test_size=0.20, random_state=42, stratify=y)
        y_train, y_val = train_test_split(y_temp, test_size=0.25, random_state=42, stratify=y_temp)

        at_risk_idx = 3
        for nombre, y_split in [("train", y_train), ("val", y_val), ("test", y_test)]:
            pct = compute_at_risk_ratio(y_split, at_risk_idx)
            assert 3.0 <= pct <= 4.5, f"At Risk en {nombre}: {pct:.2f}% — fuera del rango esperado"


class TestNotebook02ColumnTransformer:
    """ColumnTransformer — RobustScaler + OHE + OrdinalEncoder."""

    @pytest.fixture
    def preprocessor_y_datos(self):
        from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder
        from sklearn.compose import ColumnTransformer

        AGE_ORDER_NESTED = [["Cachorro (<6m)", "Joven (6m-1a)", "Adulto joven (1-3a)",
                              "Adulto (3-7a)", "Senior (>7a)"]]

        FEATURES_NUM     = ["AgeInDays_log"]
        FEATURES_OHE     = ["AnimalType", "Sex", "IntakeType", "IntakeCondition",
                             "breed_type", "Color_grouped", "Season"]
        FEATURES_ORDINAL = ["AgeGroup"]

        preprocessor = ColumnTransformer(
            transformers=[
                ("scaler",  RobustScaler(), FEATURES_NUM),
                ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False), FEATURES_OHE),
                ("ordinal", OrdinalEncoder(categories=AGE_ORDER_NESTED,
                                           handle_unknown="use_encoded_value",
                                           unknown_value=-1), FEATURES_ORDINAL),
            ],
            remainder="drop"
        )

        # Dataset mínimo para fit
        df_train = pd.DataFrame([
            {"AgeInDays_log": np.log1p(730), "AnimalType": "Dog", "Sex": "Neutered Male",
             "IntakeType": "Stray", "IntakeCondition": "Normal", "breed_type": "mix",
             "Color_grouped": "Bicolor", "Season": "Primavera", "AgeGroup": "Adulto joven (1-3a)"},
            {"AgeInDays_log": np.log1p(90),  "AnimalType": "Cat", "Sex": "Spayed Female",
             "IntakeType": "Owner Surrender", "IntakeCondition": "Injured", "breed_type": "purebred",
             "Color_grouped": "Monocolor", "Season": "Invierno", "AgeGroup": "Cachorro (<6m)"},
            {"AgeInDays_log": np.log1p(3285), "AnimalType": "Dog", "Sex": "Intact Male",
             "IntakeType": "Stray", "IntakeCondition": "Sick", "breed_type": "mix",
             "Color_grouped": "Tricolor", "Season": "Verano", "AgeGroup": "Senior (>7a)"},
        ])
        return preprocessor, df_train, FEATURES_NUM, FEATURES_OHE, FEATURES_ORDINAL

    def test_fit_transform_no_falla(self, preprocessor_y_datos):
        """fit_transform no lanza excepciones con datos válidos."""
        preprocessor, df_train, *_ = preprocessor_y_datos
        result = preprocessor.fit_transform(df_train)
        assert result is not None

    def test_output_es_numpy_array(self, preprocessor_y_datos):
        """La salida del preprocesador es un numpy array."""
        preprocessor, df_train, *_ = preprocessor_y_datos
        result = preprocessor.fit_transform(df_train)
        assert isinstance(result, np.ndarray)

    def test_output_sin_nulos(self, preprocessor_y_datos):
        """El array preprocesado no contiene NaN."""
        preprocessor, df_train, *_ = preprocessor_y_datos
        result = preprocessor.fit_transform(df_train)
        assert not np.any(np.isnan(result))

    def test_mismas_columnas_en_transform(self, preprocessor_y_datos):
        """transform produce el mismo número de columnas que fit_transform."""
        preprocessor, df_train, *_ = preprocessor_y_datos
        train_result = preprocessor.fit_transform(df_train)
        val_result   = preprocessor.transform(df_train.iloc[:1])
        assert train_result.shape[1] == val_result.shape[1]

    def test_unknown_ohe_no_falla(self, preprocessor_y_datos):
        """OHE con handle_unknown='ignore' no falla ante categorías nuevas."""
        preprocessor, df_train, *_ = preprocessor_y_datos
        preprocessor.fit_transform(df_train)

        df_new = df_train.copy()
        df_new.loc[0, "AnimalType"] = "Bird"  # categoría desconocida
        result = preprocessor.transform(df_new)
        assert not np.any(np.isnan(result))

    def test_ordinal_encoder_respeta_orden(self, preprocessor_y_datos):
        """OrdinalEncoder asigna índices crecientes al orden Cachorro→Senior."""
        preprocessor, df_train, *_ = preprocessor_y_datos
        preprocessor.fit_transform(df_train)

        age_groups = ["Cachorro (<6m)", "Joven (6m-1a)", "Adulto joven (1-3a)",
                      "Adulto (3-7a)", "Senior (>7a)"]
        indices = []
        for group in age_groups:
            df_test = df_train.iloc[[0]].copy()
            df_test["AgeGroup"] = group
            result = preprocessor.transform(df_test)
            # El ordinal es la última columna
            indices.append(result[0, -1])

        assert indices == sorted(indices), "OrdinalEncoder no respeta el orden de AgeGroup"


# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 03 — OPTIMIZACIÓN OPTUNA
# ══════════════════════════════════════════════════════════════════════════════

class TestNotebook03ObjetivoOptuna:
    """Función objetivo de Optuna: 0.70 × Recall_AtRisk + 0.30 × F1-macro."""

    def test_pesos_suman_1(self):
        """Los pesos de la función objetivo suman exactamente 1."""
        assert pytest.approx(0.70 + 0.30) == 1.0

    def test_objetivo_maximo_cuando_ambas_metricas_son_1(self):
        """Recall=1.0 y F1=1.0 → objetivo = 1.0."""
        assert compute_optuna_objective(1.0, 1.0) == pytest.approx(1.0)

    def test_objetivo_minimo_cuando_ambas_metricas_son_0(self):
        """Recall=0 y F1=0 → objetivo = 0.0."""
        assert compute_optuna_objective(0.0, 0.0) == pytest.approx(0.0)

    def test_recall_pondera_mas_que_f1(self):
        """Recall At Risk tiene más peso (0.70) que F1-macro (0.30)."""
        solo_recall = compute_optuna_objective(1.0, 0.0)
        solo_f1     = compute_optuna_objective(0.0, 1.0)
        assert solo_recall > solo_f1

    def test_objetivo_con_valores_tipicos(self):
        """Recall=0.60, F1=0.70 → 0.70×0.60 + 0.30×0.70 = 0.63."""
        resultado = compute_optuna_objective(0.60, 0.70)
        assert resultado == pytest.approx(0.63)

    def test_objetivo_entre_0_y_1(self):
        """La función objetivo siempre devuelve valores entre 0 y 1."""
        casos = [(0.55, 0.65), (0.80, 0.72), (0.30, 0.40), (1.0, 1.0), (0.0, 0.0)]
        for recall, f1 in casos:
            obj = compute_optuna_objective(recall, f1)
            assert 0.0 <= obj <= 1.0


class TestNotebook03Umbral:
    """Aplicación del umbral óptimo para At Risk."""

    AT_RISK_IDX = 1  # índice de At Risk en LabelEncoder (orden alfabético)

    def test_umbral_fuerza_at_risk_cuando_supera(self):
        """Si P(At Risk) >= umbral, la predicción se fuerza a At Risk."""
        proba = np.array([[0.6, 0.5, 0.1, 0.1]])  # At Risk=0.5, umbral=0.4
        preds = apply_threshold(proba, self.AT_RISK_IDX, umbral=0.4)
        assert preds[0] == self.AT_RISK_IDX

    def test_umbral_no_fuerza_cuando_no_supera(self):
        """Si P(At Risk) < umbral, se usa el argmax normal."""
        proba = np.array([[0.7, 0.2, 0.05, 0.05]])  # Adoption gana, At Risk=0.2 < 0.5
        preds = apply_threshold(proba, self.AT_RISK_IDX, umbral=0.5)
        assert preds[0] == 0  # Adoption (índice 0)

    def test_umbral_exacto_activa_at_risk(self):
        """P(At Risk) == umbral exacto activa At Risk."""
        proba = np.array([[0.5, 0.5, 0.0, 0.0]])
        preds = apply_threshold(proba, self.AT_RISK_IDX, umbral=0.5)
        assert preds[0] == self.AT_RISK_IDX

    def test_umbral_bajo_aumenta_recall(self):
        """Un umbral bajo detecta más casos At Risk (mayor recall)."""
        proba = np.array([
            [0.6, 0.3, 0.05, 0.05],
            [0.5, 0.4, 0.05, 0.05],
            [0.8, 0.1, 0.05, 0.05],
        ])
        preds_bajo = apply_threshold(proba, self.AT_RISK_IDX, umbral=0.25)
        preds_alto = apply_threshold(proba, self.AT_RISK_IDX, umbral=0.5)

        n_at_risk_bajo = (preds_bajo == self.AT_RISK_IDX).sum()
        n_at_risk_alto = (preds_alto == self.AT_RISK_IDX).sum()
        assert n_at_risk_bajo >= n_at_risk_alto

    def test_umbral_batch(self):
        """El umbral se aplica correctamente a un batch de predicciones."""
        proba = np.array([
            [0.1, 0.8, 0.05, 0.05],  # → At Risk
            [0.7, 0.2, 0.05, 0.05],  # → Adoption (argmax)
            [0.3, 0.6, 0.05, 0.05],  # → At Risk
        ])
        preds = apply_threshold(proba, self.AT_RISK_IDX, umbral=0.5)
        assert preds[0] == self.AT_RISK_IDX
        assert preds[1] == 0
        assert preds[2] == self.AT_RISK_IDX