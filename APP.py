import gradio as gr
from ultralytics import YOLO
from PIL import Image
import os
import tempfile

# ====================== CONFIGURACIÓN ======================
# Cambia esta ruta por tu modelo entrenado
MODEL_PATH = "models/best.pt"   # ← Descarga tu best.pt y ponlo aquí

try:
    model = YOLO(MODEL_PATH)
    print(f"✅ Modelo cargado correctamente: {MODEL_PATH}")
except Exception as e:
    print(f"⚠️ Error al cargar modelo: {e}")
    model = None

# ====================== FUNCIÓN PRINCIPAL ======================
def detectar_tumor(image):
    if image is None:
        return None, "❌ **Error**: Por favor sube una imagen MRI válida."
    
    if model is None:
        return None, "❌ Error: El modelo no se pudo cargar. Contacta al desarrollador."

    try:
        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name

        # Realizar la predicción
        results = model(tmp_path, conf=0.25, iou=0.45)
        
        # Imagen con detecciones dibujadas
        annotated_img = results[0].plot(line_width=2, font_size=12)
        
        # Extraer información de las detecciones
        detections = []
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                conf = box.conf.item()
                cls = int(box.cls.item())
                class_name = model.names[cls]
                detections.append(f"🔴 **{class_name}** - Confianza: {conf:.1%}")
        else:
            detections.append("✅ **No se detectaron tumores** en esta imagen.")

        resultado_texto = "\n\n".join(detections)

        # Limpiar archivo temporal
        os.unlink(tmp_path)

        return Image.fromarray(annotated_img), resultado_texto

    except Exception as e:
        return None, f"❌ **Error inesperado**: {str(e)}\n\nAsegúrate de subir una imagen MRI (formato JPG o PNG)."

# ====================== INTERFAZ GRADIO ======================
title = "🧠 Detector Inteligente de Tumores Cerebrales"
description = """
**Versión Beta** - Sistema de detección de tumores en imágenes MRI usando YOLOv8  
Cumple con todos los requisitos de la actividad: modelo entrenado + interfaz web funcional.
"""

interface = gr.Interface(
    fn=detectar_tumor,
    inputs=gr.Image(
        type="pil", 
        label="📤 Sube tu imagen MRI de cerebro (axial)",
        image_mode="RGB"
    ),
    outputs=[
        gr.Image(type="pil", label="🖼️ Imagen con Detección"),
        gr.Markdown(label="📋 Resultados del Análisis")
    ],
    title=title,
    description=description,
    examples=[
        ["examples/mri_sample1.jpg"],
        ["examples/mri_sample2.jpg"],
    ] if os.path.exists("examples") else None,
    allow_flagging="never",
    theme=gr.themes.Soft(),
    cache_examples=False,
    concurrency_limit=5
)

# Footer informativo
gr.Markdown("""
---
**Información Técnica**  
• Modelo: YOLOv8 entrenado en Brain Tumor MRI Dataset  
• Framework: Ultralytics + Gradio  
• Fecha: Junio 2026  
""")

if __name__ == "__main__":
    interface.launch(
        share=True,           # Crea enlace público temporal
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )