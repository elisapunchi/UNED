"""
App web: clasificador de argumentos (Universidad de la Educación - Uruguay)
----------------------------------------------------------------------------
Ahora acepta CUALQUIER tipo de archivo (PDF, Word, TXT, CSV, Excel):

- Si es un CSV/Excel que ya tiene la tabla de argumentos (columnas
  "Persona/Institución", "Postura", etc.), la usa directamente.
- Si es cualquier otro archivo (o una tabla sin esas columnas), primero
  extrae el texto y usa un LLM para armar la tabla de argumentos
  (equivalente al PROMPT_EXTRACCION del notebook original).

Luego limpia los datos y clasifica cada argumento en categorías temáticas.

Cómo correrla localmente:
    pip install -r requirements.txt
    streamlit run app.py

Cómo publicarla gratis (URL pública):
    1. Subí este archivo + requirements.txt a un repo de GitHub.
    2. Entrá a https://share.streamlit.io , conectá el repo y desplegá.
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
    "Subí cualquier archivo (PDF, Word, TXT, CSV o Excel) con el texto a analizar. "
    "La app extrae los argumentos, los limpia y los clasifica automáticamente con un LLM."
)

COLUMNAS_TEXTO = [
    "Persona/Institución",
    "A quién representa",
    "Postura",
    "Argumento (resumen)",
    "Palabras clave",
]

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

# Tamaño de cada bloque de texto que se envía al LLM en el paso de extracción
TAMANO_BLOQUE = 6000

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
        "Modelo para clasificar/extraer",
        ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
        index=0,
    )

    st.subheader("Tema del análisis")
    tema = st.text_area(
        "Describí brevemente sobre qué tema hay que buscar argumentos a favor/en contra",
        value=(
            "la creación de una Universidad de la Educación en Uruguay, institución que "
            "tendría a su cargo la formación de maestros, profesores y educadores sociales"
        ),
        height=80,
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
    )

# ----------------------------------------------------------------------------
# Extracción de texto según el tipo de archivo
# ----------------------------------------------------------------------------
def leer_pdf(archivo) -> str:
    from pypdf import PdfReader

    reader = PdfReader(archivo)
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def leer_docx(archivo) -> str:
    import docx

    documento = docx.Document(archivo)
    return "\n\n".join(p.text for p in documento.paragraphs if p.text.strip())


def leer_txt(archivo) -> str:
    return archivo.read().decode("utf-8", errors="ignore")


def dividir_en_bloques(texto: str, tamano: int = TAMANO_BLOQUE):
    """Divide el texto en bloques por párrafos, sin cortar oraciones a la mitad."""
    parrafos = texto.split("\n")
    bloques, actual = [], ""
    for parrafo in parrafos:
        if len(actual) + len(parrafo) > tamano and actual:
            bloques.append(actual)
            actual = ""
        actual += parrafo + "\n"
    if actual.strip():
        bloques.append(actual)
    return bloques


# ----------------------------------------------------------------------------
# Paso 1: subir archivo
# ----------------------------------------------------------------------------
archivo = st.file_uploader(
    "Subí tu archivo (PDF, Word, TXT, CSV o Excel)",
    type=["pdf", "docx", "txt", "csv", "xlsx", "xls"],
)

if archivo is None:
    st.info("Esperando que subas un archivo para empezar.")
    st.stop()

nombre = archivo.name.lower()
df_crudo = None
texto_bruto = None

if nombre.endswith((".csv", ".xlsx", ".xls")):
    try:
        if nombre.endswith(".csv"):
            df_tabla = pd.read_csv(archivo)
        else:
            df_tabla = pd.read_excel(archivo)
    except Exception as e:
        st.error(f"No pude leer el archivo: {e}")
        st.stop()

    if all(col in df_tabla.columns for col in COLUMNAS_TEXTO):
        st.success("El archivo ya tiene la tabla de argumentos lista. Se usará directamente.")
        df_crudo = df_tabla
    else:
        st.info(
            "El archivo es una tabla pero no tiene las columnas esperadas: se va a tratar "
            "su contenido como texto y extraer los argumentos con el LLM."
        )
        texto_bruto = df_tabla.to_string(index=False)
elif nombre.endswith(".pdf"):
    texto_bruto = leer_pdf(archivo)
elif nombre.endswith(".docx"):
    texto_bruto = leer_docx(archivo)
elif nombre.endswith(".txt"):
    texto_bruto = leer_txt(archivo)

if texto_bruto is not None and not texto_bruto.strip():
    st.error("No pude extraer texto de ese archivo. Probá con otro formato.")
    st.stop()

if texto_bruto is not None:
    with st.expander("Ver texto extraído del archivo"):
        st.text(texto_bruto[:5000] + ("..." if len(texto_bruto) > 5000 else ""))

# ----------------------------------------------------------------------------
# Botón principal: corre todo el pipeline
# ----------------------------------------------------------------------------
if not api_key:
    st.warning("Ingresá tu OpenAI API key en la barra lateral para continuar.")
    st.stop()

if not CATEGORIAS:
    st.warning("Definí al menos una categoría en la barra lateral.")
    st.stop()

if st.button("🚀 Analizar documento", type="primary"):
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    # ---------------- Paso "Bajar": extracción de argumentos (si hace falta) ----
    if df_crudo is None:
        bloques = dividir_en_bloques(texto_bruto)
        st.write(f"Extrayendo argumentos de {len(bloques)} bloque(s) de texto...")
        progreso = st.progress(0.0, text="Extrayendo argumentos...")
        filas_extraidas = []

        prompt_extraccion = """
Sos un asistente especializado en análisis documental. Tu tarea es construir una base de
datos con los argumentos a favor y en contra de {tema}.

Reglas de trabajo:
- Trabajá únicamente con el texto provisto abajo. No busques información en internet ni
  completes con conocimiento externo.
- Si el texto no contiene argumentos claros sobre el tema, devolvé una lista vacía.
- Cada argumento debe basarse en lo que el texto dice explícitamente.
- Si un mismo párrafo contiene varios argumentos distintos, generá una fila por cada uno.
- Distinguí entre argumentos sustantivos y menciones estratégicas/tácticas; incluí solo
  los sustantivos.

Devolvé un JSON con la clave "argumentos": una lista de objetos, cada uno con estos
campos exactos:
- "Persona/Institución"
- "A quién representa"
- "Postura" (exactamente "A favor" o "En contra")
- "Argumento (resumen)"
- "Palabras clave"
- "Fuente/cita en el documento"

Texto a analizar:
\"\"\"
{bloque}
\"\"\"

Devolvé solo el JSON, sin texto adicional.
"""

        for i, bloque in enumerate(bloques):
            try:
                response = client.chat.completions.create(
                    model=modelo,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt_extraccion.format(tema=tema, bloque=bloque),
                        }
                    ],
                    response_format={"type": "json_object"},
                )
                data = json.loads(response.choices[0].message.content)
                filas_extraidas.extend(data.get("argumentos", []))
            except Exception as e:
                st.warning(f"Bloque {i + 1}: no se pudo procesar ({e}).")
            progreso.progress((i + 1) / len(bloques), text=f"Extrayendo... {i + 1}/{len(bloques)}")

        progreso.empty()

        if not filas_extraidas:
            st.error(
                "No se encontraron argumentos en el documento. Probá con otro archivo o "
                "revisá la descripción del tema."
            )
            st.stop()

        df_crudo = pd.DataFrame(filas_extraidas)
        st.success(f"Se extrajeron {len(df_crudo)} argumentos del documento.")

    faltantes = [c for c in COLUMNAS_TEXTO if c not in df_crudo.columns]
    if faltantes:
        st.error("Al resultado le faltan columnas: " + ", ".join(faltantes))
        st.stop()

    # ---------------- Paso "Limpiar" --------------------------------------
    df = df_crudo.copy()
    for col in COLUMNAS_TEXTO:
        df[col] = df[col].astype(str).str.strip()
    df = df[df["Postura"].isin(["A favor", "En contra"])]
    df = df.dropna(subset=["Argumento (resumen)"])
    df = df[df["Argumento (resumen)"] != ""]
    df = df.reset_index(drop=True)

    st.write(f"✅ Quedaron **{len(df)}** argumentos después de limpiar.")

    if len(df) == 0:
        st.error("No quedaron argumentos válidos después de limpiar los datos.")
        st.stop()

    # ---------------- Paso "Procesar": clasificación temática --------------
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

    st.success("Análisis completo ✅")
    st.dataframe(df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Argumentos por categoría**")
        st.bar_chart(df["Categoria"].value_counts())
    with col2:
        st.write("**Argumentos por postura**")
        st.bar_chart(df["Postura"].value_counts())

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

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="⬇️ Descargar CSV con resultados",
        data=csv_buffer.getvalue(),
        file_name="argumentos_categorizados.csv",
        mime="text/csv",
    )
