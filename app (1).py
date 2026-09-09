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
    "Subí uno o varios archivos (PDF, Word, TXT, CSV o Excel) con el texto a analizar. "
    "La app extrae los argumentos de todos ellos, los junta, los limpia y los clasifica "
    "automáticamente con un LLM."
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
# Resumen escrito: cálculo de estadísticas y armado del Word
# ----------------------------------------------------------------------------
def calcular_resumen(df: pd.DataFrame) -> dict:
    """Calcula las tablas y conclusiones que después se muestran en pantalla y en el Word."""
    resumen = {"total": len(df), "conteo_postura": df["Postura"].value_counts()}

    if "Categoria" in df.columns:
        ct = pd.crosstab(df["Categoria"], df["Postura"])
        ct["Total"] = ct.sum(axis=1)
        resumen["categoria_postura"] = ct.sort_values("Total", ascending=False)

    if "Persona/Institución" in df.columns:
        resumen["top_personas"] = df["Persona/Institución"].value_counts().head(8)

    if "Archivo fuente" in df.columns and df["Archivo fuente"].nunique() > 1:
        resumen["conteo_archivos"] = df["Archivo fuente"].value_counts()

    # ---- Conclusiones automáticas, en texto plano ----
    total = resumen["total"]
    cp = resumen["conteo_postura"]
    conclusiones = []

    postura_top = cp.idxmax()
    conclusiones.append(
        f"La mayoría de los argumentos relevados están {postura_top.lower()} "
        f"({cp.max() / total * 100:.0f}% del total)."
    )

    if "categoria_postura" in resumen:
        ct = resumen["categoria_postura"]
        cat_top = ct["Total"].idxmax()
        conclusiones.append(
            f'La categoría más discutida es "{cat_top}", con {ct.loc[cat_top, "Total"]} '
            f"argumentos ({ct.loc[cat_top, 'Total'] / total * 100:.0f}% del total)."
        )

        if "A favor" in ct.columns and "En contra" in ct.columns:
            candidatos = ct[ct["Total"] >= 2].copy()
            if not candidatos.empty:
                candidatos["pct_favor"] = candidatos["A favor"] / candidatos["Total"]
                cat_favor = candidatos["pct_favor"].idxmax()
                if candidatos.loc[cat_favor, "pct_favor"] > 0.5:
                    conclusiones.append(
                        f'"{cat_favor}" es la categoría con mayor proporción de argumentos a favor.'
                    )
                cat_contra = candidatos["pct_favor"].idxmin()
                if candidatos.loc[cat_contra, "pct_favor"] < 0.5:
                    conclusiones.append(
                        f'"{cat_contra}" es la categoría con mayor proporción de argumentos en contra.'
                    )

    if "top_personas" in resumen:
        actor_top = resumen["top_personas"].idxmax()
        conclusiones.append(
            f"El actor con más argumentos registrados es {actor_top}, "
            f"con {resumen['top_personas'].max()}."
        )

    resumen["conclusiones"] = conclusiones
    return resumen


def mostrar_resumen_en_pantalla(df: pd.DataFrame, resumen: dict) -> None:
    st.write(f"**Total de argumentos analizados:** {resumen['total']}")
    for c in resumen["conclusiones"]:
        st.markdown(f"- {c}")

    with st.expander("Ver tablas del resumen"):
        st.write("**Por postura**")
        st.table(resumen["conteo_postura"])
        if "categoria_postura" in resumen:
            st.write("**Por categoría**")
            st.table(resumen["categoria_postura"])
        if "top_personas" in resumen:
            st.write("**Actores más presentes**")
            st.table(resumen["top_personas"])
        if "conteo_archivos" in resumen:
            st.write("**Por documento fuente**")
            st.table(resumen["conteo_archivos"])


def construir_resumen_docx(df: pd.DataFrame, resumen: dict) -> io.BytesIO:
    from docx import Document
    from docx.shared import Pt

    doc = Document()
    doc.styles["Normal"].font.name = "Arial"
    doc.styles["Normal"].font.size = Pt(10)

    doc.add_heading("Resumen de argumentos analizados", level=0)

    n_docs = df["Archivo fuente"].nunique() if "Archivo fuente" in df.columns else None
    subtitulo = f"Análisis de {resumen['total']} argumentos"
    if n_docs:
        subtitulo += f" extraídos de {n_docs} documento(s)"
    parrafo = doc.add_paragraph()
    parrafo.add_run(subtitulo).italic = True

    def agregar_tabla(titulo, serie_o_df, nombre_columna_indice):
        doc.add_heading(titulo, level=1)
        if isinstance(serie_o_df, pd.Series):
            tabla = doc.add_table(rows=1, cols=2)
            tabla.style = "Light Grid Accent 1"
            hdr = tabla.rows[0].cells
            hdr[0].text, hdr[1].text = nombre_columna_indice, "Cantidad"
            for indice, valor in serie_o_df.items():
                fila = tabla.add_row().cells
                fila[0].text = str(indice)
                fila[1].text = str(valor)
        else:
            columnas = list(serie_o_df.columns)
            tabla = doc.add_table(rows=1, cols=len(columnas) + 1)
            tabla.style = "Light Grid Accent 1"
            hdr = tabla.rows[0].cells
            hdr[0].text = nombre_columna_indice
            for i, col in enumerate(columnas):
                hdr[i + 1].text = str(col)
            for indice, fila_datos in serie_o_df.iterrows():
                fila = tabla.add_row().cells
                fila[0].text = str(indice)
                for i, col in enumerate(columnas):
                    fila[i + 1].text = str(fila_datos[col])

    agregar_tabla("1. Panorama general (por postura)", resumen["conteo_postura"], "Postura")

    if "categoria_postura" in resumen:
        agregar_tabla(
            "2. Distribución por categoría temática", resumen["categoria_postura"], "Categoría"
        )

    if "top_personas" in resumen:
        agregar_tabla("3. Voces más presentes", resumen["top_personas"], "Persona/Institución")

    if "conteo_archivos" in resumen:
        agregar_tabla("4. Documentos fuente", resumen["conteo_archivos"], "Documento")

    doc.add_heading("Conclusiones", level=1)
    for c in resumen["conclusiones"]:
        doc.add_paragraph(c, style="List Bullet")

    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer


# ----------------------------------------------------------------------------
# Paso 1: subir archivo(s)
# ----------------------------------------------------------------------------
archivos = st.file_uploader(
    "Subí uno o más archivos (PDF, Word, TXT, CSV o Excel)",
    type=["pdf", "docx", "txt", "csv", "xlsx", "xls"],
    accept_multiple_files=True,
)

if not archivos:
    st.info("Esperando que subas al menos un archivo para empezar.")
    st.stop()

st.write(f"📎 {len(archivos)} archivo(s) subido(s).")

# Cada elemento queda como: {"nombre", "tipo": "tabla"|"texto", "contenido": df o str}
documentos = []

for archivo in archivos:
    nombre = archivo.name.lower()

    if nombre.endswith((".csv", ".xlsx", ".xls")):
        try:
            if nombre.endswith(".csv"):
                df_tabla = pd.read_csv(archivo)
            else:
                df_tabla = pd.read_excel(archivo)
        except Exception as e:
            st.error(f"No pude leer '{archivo.name}': {e}")
            continue

        if all(col in df_tabla.columns for col in COLUMNAS_TEXTO):
            st.success(f"'{archivo.name}': ya tiene la tabla de argumentos lista, se usará directamente.")
            documentos.append({"nombre": archivo.name, "tipo": "tabla", "contenido": df_tabla})
        else:
            st.info(
                f"'{archivo.name}': es una tabla sin las columnas esperadas, se tratará como texto."
            )
            documentos.append(
                {"nombre": archivo.name, "tipo": "texto", "contenido": df_tabla.to_string(index=False)}
            )
    elif nombre.endswith(".pdf"):
        documentos.append({"nombre": archivo.name, "tipo": "texto", "contenido": leer_pdf(archivo)})
    elif nombre.endswith(".docx"):
        documentos.append({"nombre": archivo.name, "tipo": "texto", "contenido": leer_docx(archivo)})
    elif nombre.endswith(".txt"):
        documentos.append({"nombre": archivo.name, "tipo": "texto", "contenido": leer_txt(archivo)})

# Descartar documentos de texto vacíos
documentos_validos = []
for doc in documentos:
    if doc["tipo"] == "texto" and not doc["contenido"].strip():
        st.warning(f"'{doc['nombre']}': no se pudo extraer texto, se va a ignorar.")
        continue
    documentos_validos.append(doc)
documentos = documentos_validos

if not documentos:
    st.error("No quedó ningún archivo válido para analizar.")
    st.stop()

with st.expander(f"Ver texto extraído ({sum(1 for d in documentos if d['tipo'] == 'texto')} archivo(s) de texto)"):
    for doc in documentos:
        if doc["tipo"] == "texto":
            st.markdown(f"**{doc['nombre']}**")
            contenido = doc["contenido"]
            st.text(contenido[:3000] + ("..." if len(contenido) > 3000 else ""))

# ----------------------------------------------------------------------------
# Botón principal: corre todo el pipeline
# ----------------------------------------------------------------------------
if not api_key:
    st.warning("Ingresá tu OpenAI API key en la barra lateral para continuar.")
    st.stop()

if not CATEGORIAS:
    st.warning("Definí al menos una categoría en la barra lateral.")
    st.stop()

if st.button("🚀 Analizar documentos", type="primary"):
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

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

    # ---------------- Paso "Bajar": extracción de argumentos por documento ----
    piezas_df = []

    documentos_texto = [d for d in documentos if d["tipo"] == "texto"]
    documentos_tabla = [d for d in documentos if d["tipo"] == "tabla"]

    for doc in documentos_tabla:
        df_pieza = doc["contenido"].copy()
        df_pieza["Archivo fuente"] = doc["nombre"]
        piezas_df.append(df_pieza)

    if documentos_texto:
        # Armamos todos los bloques de todos los documentos de texto en una sola lista,
        # cada uno recordando de qué archivo vino, para poder mostrar progreso conjunto.
        bloques_con_origen = []
        for doc in documentos_texto:
            for bloque in dividir_en_bloques(doc["contenido"]):
                bloques_con_origen.append((doc["nombre"], bloque))

        st.write(
            f"Extrayendo argumentos de {len(documentos_texto)} documento(s) "
            f"({len(bloques_con_origen)} bloque(s) de texto en total)..."
        )
        progreso = st.progress(0.0, text="Extrayendo argumentos...")
        filas_extraidas = []
        errores_extraccion = []

        for i, (nombre_doc, bloque) in enumerate(bloques_con_origen):
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
                for fila in data.get("argumentos", []):
                    fila["Archivo fuente"] = nombre_doc
                    filas_extraidas.append(fila)
            except Exception as e:
                mensaje = f"'{nombre_doc}', bloque {i + 1}: no se pudo procesar ({e})."
                st.warning(mensaje)
                errores_extraccion.append(mensaje)
            progreso.progress(
                (i + 1) / len(bloques_con_origen),
                text=f"Extrayendo... {i + 1}/{len(bloques_con_origen)}",
            )

        progreso.empty()

        if filas_extraidas:
            piezas_df.append(pd.DataFrame(filas_extraidas))
    else:
        errores_extraccion = []

    if not piezas_df:
        detalle = ""
        if errores_extraccion:
            ejemplos = "\n".join(f"- {e}" for e in errores_extraccion[:5])
            detalle = (
                f"\n\nDetalle de los errores encontrados (mostrando hasta 5 de "
                f"{len(errores_extraccion)}):\n{ejemplos}\n\n"
                "Si el mensaje menciona 'API key', 'quota', 'rate limit' o "
                "'model', revisá tu clave de OpenAI o el modelo elegido en la "
                "barra lateral: eso suele ser la causa cuando el mismo archivo "
                "funcionó antes sin problema."
            )
        st.error(
            "No se encontraron argumentos en ninguno de los documentos. Probá con otros "
            "archivos o revisá la descripción del tema." + detalle
        )
        st.stop()

    df_crudo = pd.concat(piezas_df, ignore_index=True)
    st.success(f"Se juntaron {len(df_crudo)} argumentos de {len(documentos)} archivo(s).")

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

    st.subheader("📝 Resumen escrito")
    resumen = calcular_resumen(df)
    mostrar_resumen_en_pantalla(df, resumen)

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Argumentos")
    excel_buffer.seek(0)

    resumen_buffer = construir_resumen_docx(df, resumen)

    col_desc1, col_desc2, col_desc3 = st.columns(3)
    with col_desc1:
        st.download_button(
            label="⬇️ Descargar Excel con resultados",
            data=excel_buffer,
            file_name="argumentos_categorizados.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with col_desc2:
        st.download_button(
            label="⬇️ Descargar resumen (Word)",
            data=resumen_buffer,
            file_name="resumen_argumentos.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    with col_desc3:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Descargar CSV con resultados",
            data=csv_buffer.getvalue(),
            file_name="argumentos_categorizados.csv",
            mime="text/csv",
        )
