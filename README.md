# Dynamic Incident Response System (Hybrid ML Approach)
## Sistema Dinámico de Respuesta ante Incidentes (Enfoque ML Híbrido)

### 🚨 Overview / Resumen
**English:** This project implements a real-time decision engine for network intrusion detection. It dynamically selects between **LightGBM** (for speed/low latency) and **CatBoost** (for deep analysis) based on server load and asset criticality.
**Español:** Este proyecto implementa un motor de decisión en tiempo real para la detección de intrusiones en red. Selecciona dinámicamente entre **LightGBM** (velocidad) y **CatBoost** (análisis profundo) según la carga del servidor y la importancia del activo.

### 🛠️ Tech Stack / Tecnologías
- **Language:** Python 3.12
- **ML Libraries:** Pandas, Scikit-Learn, LightGBM, CatBoost
- **Dataset:** CIC-IDS2017 (Consolidated network logs)

### 📈 Key Results / Resultados Clave
- **High Accuracy:** ~99.2% overall accuracy.
- **Efficiency:** LightGBM proved to be 15% more effective at reducing False Negatives in this specific environment.
- **Scalability:** Processed nearly 400,000 network records from 8 different attack scenarios.

### 🧠 Logic / Lógica de Decisión
```python
if asset_criticality >= 8 and cpu_load < 0.6:
    use_model("CatBoost") # Deep Inspection
else:
    use_model("LightGBM") # Rapid Response

    Contact: Gustavo Maldonado |
Email: gustavo.a.maldonado.v@gmail.com |