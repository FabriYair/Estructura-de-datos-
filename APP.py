import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import tempfile

# ====================== CONFIGURACIÓN ======================

MODEL_PATH = "models/best.pt"

st.set_page_config(
    page_title="Detector Inteligente de Tumores",
    page_icon="🧠",
    layout="centered"
)

# ====================== CARGA DEL MODELO ======================

@st.cache_resource
def cargar_modelo():
    """
    Carga el modelo YOLO entrenado.
    """
    if not os.path.exists(MODEL_PATH):
        return None

    try:
        modelo = YOLO(MODEL_PATH)
        return modelo
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None


modelo = cargar_modelo()

# ====================== INTERFAZ PRINCIPAL ======================

st.title("🧠 Detector Inteligente de Tumores Cerebrales")

st.subheader("Herramienta de apoyo para la detección temprana mediante Inteligencia Artificial")

st.write(
    """
    Esta versión Beta permite cargar una imagen médica MRI y analizarla mediante un modelo
    de Inteligencia Artificial entrenado para detectar posibles tumores cerebrales.
    """
)

st.warning(
    "⚠️ Este sistema es una herramienta de apoyo académico y tecnológico. "
    "No reemplaza el diagnóstico médico profesional."
)

# ====================== VERIFICACIÓN DEL MODELO ======================

if modelo is None:
    st.error(
        "❌ No se encontró el modelo entrenado. "
        "Verifica que el archivo `best.pt` esté en la carpeta `models/`."
    )
else:
    st.success("✅ Modelo cargado correctamente.")

# ====================== CARGA DE IMAGEN ======================

st.header("📤 Cargar imagen MRI")

archivo = st.file_uploader(
    "Sube una imagen médica en formato JPG, JPEG o PNG",
    type=["jpg", "jpeg", "png"]
)

if archivo is None:
    st.info("📌 Sube una imagen para iniciar el análisis.")

else:
    try:
        imagen = Image.open(archivo).convert("RGB")

        st.subheader("🖼️ Imagen cargada")
        st.image(imagen, caption="Imagen MRI cargada por el usuario", use_container_width=True)

        if st.button("🔍 Analizar imagen"):

            if modelo is None:
                st.error("❌ No se puede analizar la imagen porque el modelo no fue cargado.")

            else:
                try:
                    with st.spinner("Analizando imagen con el modelo de IA..."):

                        # Guardar la imagen temporalmente para que YOLO pueda procesarla
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                            imagen.save(tmp.name)
                            tmp_path = tmp.name

                        # Realizar predicción
                        resultados = modelo(tmp_path, conf=0.25, iou=0.45)

                        # Imagen con detecciones dibujadas
                        imagen_anotada = resultados[0].plot(line_width=2)

                        detecciones = []

                        if resultados[0].boxes is not None and len(resultados[0].boxes) > 0:
                            for box in resultados[0].boxes:
                                confianza = box.conf.item()
                                clase = int(box.cls.item())
                                nombre_clase = modelo.names[clase]

                                detecciones.append(
                                    f"🔴 **{nombre_clase}** - Confianza: {confianza:.1%}"
                                )
                        else:
                            detecciones.append("✅ **No se detectaron tumores en esta imagen.**")

                        # Eliminar archivo temporal
                        os.unlink(tmp_path)

                    st.subheader("🖼️ Imagen con detección")
                    st.image(imagen_anotada, caption="Resultado del análisis", use_container_width=True)

                    st.subheader("📋 Resultados del análisis")

                    for deteccion in detecciones:
                        st.markdown(deteccion)

                    st.info(
                        "Este resultado debe ser interpretado como apoyo y no como diagnóstico médico final."
                    )

                except Exception as e:
                    st.error("❌ Ocurrió un error durante la predicción.")
                    st.code(str(e))

    except Exception as e:
        st.error("❌ El archivo cargado no corresponde a una imagen válida.")
        st.code(str(e))

# ====================== MÉTRICAS DEL MODELO ======================

st.markdown("---")
st.header("📊 Métricas del modelo")

st.write(
    """
    Las siguientes métricas pueden ser actualizadas por el equipo según los resultados
    obtenidos durante la evaluación del modelo.
    """
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", "Pendiente")
col2.metric("Precision", "Pendiente")
col3.metric("Recall", "Pendiente")
col4.metric("F1-Score", "Pendiente")

st.markdown("---")
st.caption("Versión Beta - Proyecto académico de Construcción de Software | Universidad Continental")
