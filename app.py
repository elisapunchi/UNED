"""
App web: clasificador de argumentos (Universidad de la Educación - Uruguay)
----------------------------------------------------------------------------
Reproduce el pipeline del notebook UNED.ipynb (Bajar -> Limpiar -> Procesar -> Guardar)
como una app web con Streamlit: el usuario sube el CSV y la app hace todo el resto.

Cómo correrla localmente:
    pip install -r requirements.txt
    streamlit run app.py

Cómo publicarla gratis (URL pública):
    1. Subí este archivo + requirements.txt a un repo de GitHub.
    2. Entrá a https://share.streamlit.io , conectá el repo y desplegá.
    3. En "Secrets" de Streamlit Cloud podés precargar OPENAI_API_KEY si querés
       evitar que cada usuario tenga que pegar su propia clave.
"""

import io
import json

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Configuración de la página
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Clasificador de argumentos", page_icon="📊", layout="wide")

st.title("📊 Clasificador de argumentos")
st.caption(
    "Subí el CSV con los argumentos (columnas: Persona/Institución, A quién representa, "
    "Postura, Argumento (resumen), Palabras clave, Fuente/cita en el documento) y la app "
    "los limpia y clasifica automáticamente con un LLM."
)

# Columnas esperadas (mismas que en el notebook original)
COLUMNAS_TEXTO = [
    "Persona/Institución",
    "A quién representa",
    "Postura",
    "Argumento (resumen)",
    "Palabras clave",
]

# Taxonomía cerrada de categorías (la misma que definiste en el notebook)
CATEGORIAS_DEFAULT = [
    "Autonomía y cogobierno",
    "Calidad académica e investigación",
    "Reconocimiento institucional y jerarquización",
    "Justicia social y democratización",
    "Viabilidad institucional y recursos",
    "Marco jurídico y consenso político",
    "Riesgos de burocratización y control democrático",
    "Acreditación docente",
    "Intereses corporativos",
    "Decentralización",
    "Inclusión educativa",
]

# ----------------------------------------------------------------------------
# Barra lateral: configuración
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuración")

    api_key = st.text_input(
        "OpenAI API key",
        type="password",
        help="Tu clave no se guarda en ningún lado; solo se usa durante esta sesión.",
    )

    modelo = st.selectbox(
        "Modelo para clasificar",
        ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
        index=0,
    )

    st.subheader("Categorías temáticas")
    categorias_texto = st.text_area(
        "Una categoría por línea (podés editarlas)",
        value="\n".join(CATEGORIAS_DEFAULT),
        height=220,
    )
    CATEGORIAS = [c.strip() for c in categorias_texto.splitlines() if c.strip()]

    hacer_clustering = st.checkbox(
        "Incluir validación por clustering (embeddings, opcional)",
        value=False,
        help="Agrupa los argumentos con un algoritmo no supervisado y los compara contra "
        "la clasificación del LLM. Usa llamadas extra a la API de embeddings.",
    )

# ----------------------------------------------------------------------------
# Paso 1: Bajar (subir archivo)
# ----------------------------------------------------------------------------
archivo = st.file_uploader("Subí tu archivo CSV", type=["csv"])

if archivo is None:
    st.info("Esperando que subas un archivo CSV para empezar.")
    st.stop()

try:
    df_crudo = pd.read_csv(archivo)
except Exception as e:
    st.error(f"No pude leer el CSV: {e}")
    st.stop()

st.success(f"Archivo cargado: {len(df_crudo)} filas, {len(df_crudo.columns)} columnas.")
with st.expander("Ver datos crudos"):
    st.dataframe(df_crudo.head(20), use_container_width=True)

faltantes = [c for c in COLUMNAS_TEXTO if c not in df_crudo.columns]
if faltantes:
    st.error(
        "Al archivo le faltan estas columnas esperadas: "
        + ", ".join(faltantes)
        + ". Revisá el CSV y volvé a subirlo."
    )
    st.stop()

# ----------------------------------------------------------------------------
# Paso 2: Limpiar
# ----------------------------------------------------------------------------
df = df_crudo.copy()

for col in COLUMNAS_TEXTO:
    df[col] = df[col].astype(str).str.strip()

df = df[df["Postura"].isin(["A favor", "En contra"])]
df = df.dropna(subset=["Argumento (resumen)"])
df = df[df["Argumento (resumen)"] != ""]
df = df.reset_index(drop=True)

st.write(f"✅ Quedaron **{len(df)}** argumentos después de limpiar.")

# ----------------------------------------------------------------------------
# Paso 3: Procesar (clasificación con LLM)
# ----------------------------------------------------------------------------
st.subheader("Clasificación")

if not api_key:
    st.warning("Ingresá tu OpenAI API key en la barra lateral para poder clasificar.")
    st.stop()

if not CATEGORIAS:
    st.warning("Definí al menos una categoría en la barra lateral.")
    st.stop()

if st.button("🚀 Clasificar argumentos", type="primary"):
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    def extraer_categoria(argumento: str, palabras_clave: str) -> str:
        prompt = f"""
        Analizá este argumento y devolvé un JSON con:

        - categoria: una de estas opciones exactas: {CATEGORIAS}

        Argumento: "{argumento}"
        Palabras clave: "{palabras_clave}"

        Solo JSON.
        """
        response = client.chat.completions.create(
            model=modelo,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        categoria = data.get("categoria", "")
        return categoria if categoria in CATEGORIAS else CATEGORIAS[0]

    progreso = st.progress(0.0, text="Clasificando argumentos...")
    categorias_resultado = []
    errores = 0

    for i, row in df.iterrows():
        try:
            categoria = extraer_categoria(row["Argumento (resumen)"], row["Palabras clave"])
        except Exception:
            categoria = None
            errores += 1
        categorias_resultado.append(categoria)
        progreso.progress((i + 1) / len(df), text=f"Clasificando... {i + 1}/{len(df)}")

    progreso.empty()
    df["Categoria"] = categorias_resultado

    if errores:
        st.warning(f"⚠️ {errores} argumento(s) no se pudieron clasificar (quedaron vacíos).")

    # Codificación numérica (igual que el notebook)
    df["Postura_num"] = df["Postura"].map({"A favor": 1, "En contra": 0})
    cat_to_num = {c: i for i, c in enumerate(CATEGORIAS)}
    df["Categoria_num"] = df["Categoria"].map(cat_to_num)
    df_onehot = pd.get_dummies(df["Categoria"], prefix="cat")
    df = pd.concat([df, df_onehot], axis=1)

    st.session_state["df_resultado"] = df

# ----------------------------------------------------------------------------
# Resultados
# ----------------------------------------------------------------------------
if "df_resultado" in st.session_state:
    df = st.session_state["df_resultado"]

    st.success("Clasificación completa ✅")
    st.dataframe(df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Argumentos por categoría**")
        st.bar_chart(df["Categoria"].value_counts())
    with col2:
        st.write("**Argumentos por postura**")
        st.bar_chart(df["Postura"].value_counts())

    # Validación opcional por clustering
    if hacer_clustering:
        st.subheader("Validación por clustering (embeddings)")
        with st.spinner("Calculando embeddings y agrupando..."):
            from openai import OpenAI
            from sklearn.cluster import KMeans

            client = OpenAI(api_key=api_key)

            def encode(textos):
                response = client.embeddings.create(
                    model="text-embedding-3-small", input=textos
                )
                return np.array([d.embedding for d in response.data])

            texto_para_embedding = (
                df["Argumento (resumen)"] + ". " + df["Palabras clave"]
            ).tolist()
            embeddings = encode(texto_para_embedding)

            kmeans = KMeans(n_clusters=len(CATEGORIAS), random_state=42, n_init=10)
            df["Cluster_embedding"] = kmeans.fit_predict(embeddings)

        tabla = pd.crosstab(df["Categoria"], df["Cluster_embedding"])
        st.write("Categoría (LLM) vs. Cluster (embeddings):")
        st.dataframe(tabla, use_container_width=True)
        st.session_state["df_resultado"] = df

    # Descarga
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="⬇️ Descargar CSV con resultados",
        data=csv_buffer.getvalue(),
        file_name="argumentos_categorizados.csv",
        mime="text/csv",
    )
