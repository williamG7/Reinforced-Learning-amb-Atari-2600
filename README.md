# Aprendizaje por Refuerzo con Atari 2600 🎮🤖

## 📋 Descripción

Proyecto educativo de **Machine Learning** que implementa un agente de **Aprendizaje por Refuerzo (RL)** para dominar juegos clásicos del **Atari 2600** usando algoritmos modernos de IA. Este repositorio es un estudio práctico completo sobre cómo entrenar agentes inteligentes desde cero.

El proyecto demuestra:
- **Entrenamiento de agentes autónomos** mediante Reinforcement Learning
- **Uso de redes neuronales convolucionales (CNN)** para procesamiento de imágenes
- **Algoritmo PPO (Proximal Policy Optimization)** para optimización de políticas
- **Técnicas avanzadas** de entrenamiento: frame stacking, reward shaping, wrappers de Atari
- **Evaluación y análisis** del rendimiento del agente entrenado

Este proyecto es ideal para:
- 📚 Estudiantes aprendiendo RL y Deep Learning
- 🔬 Investigadores en AI/ML
- 👨‍💻 Desarrolladores interesados en Game AI
- 🎯 Cualquiera que quiera comprender cómo funcionan los agentes inteligentes

---

## 🎯 Objetivos del Proyecto

1. **Comprender Reinforcement Learning** desde los conceptos fundamentales hasta técnicas avanzadas
2. **Seleccionar y caracterizar** un entorno de juego del Atari 2600
3. **Diseñar un espacio de acciones** apropiado para el juego
4. **Entrenar un agente** usando PPO y arquitectura convolucional
5. **Optimizar el rendimiento** mediante ajuste de hiperparámetros y técnicas de reward shaping
6. **Evaluar resultados** y documentar aprendizajes
7. **Analizar comportamientos** emergentes del agente entrenado

---

## 🎮 Juego Seleccionado: Space Invaders

### Descripción
**Space Invaders** es uno de los juegos más icónicos del Atari 2600. El jugador controla una nave espacial en la parte inferior de la pantalla con el objetivo de disparar a oleadas de alienígenas antes de que lleguen al suelo.

### Cómo se consigue puntuación
- 🎯 **Eliminar aliens:** Los aliens de filas superiores valen más puntos
- 🛸 **OVNI especial:** Aparece ocasionalmente y otorga puntos extra
- ⚡ **Bonus por oleadas:** Cada oleada eliminada incrementa la dificultad

### Objetivos del Juego
| Aspecto | Descripción |
|---------|-------------|
| **Ganar** | Eliminar todos los aliens antes de que lleguen al suelo |
| **Perder** | Los aliens llegan a la base o destruyen las 3 vidas |
| **Dificultad** | Aumenta con cada oleada completada |

### Acciones Disponibles

| Acción | ID | Descripción |
|--------|-----|-------------|
| NOOP | 0 | No hacer nada |
| FIRE | 1 | Disparar |
| RIGHT | 2 | Moverse a la derecha |
| LEFT | 3 | Moverse a la izquierda |
| RIGHTFIRE | 4 | Moverse derecha + disparar |
| LEFTFIRE | 5 | Moverse izquierda + disparar |

---

## 🛠️ Tecnologías y Dependencias

### Principales
```python
# Aprendizaje por Refuerzo
import gymnasium as gym              # Entorno de juegos Atari
import ale_py                        # ALE (Atari Learning Environment)
from stable_baselines3 import PPO     # Algoritmo PPO

# Procesamiento de imágenes
import numpy as np                   # Cálculos numéricos
import matplotlib.pyplot as plt      # Visualización

# Deep Learning
import torch                         # Framework neural (usado por SB3)
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
```

### Versiones Recomendadas
```
Python ≥ 3.8
gymnasium ≥ 0.28
stable-baselines3 ≥ 2.0
torch ≥ 2.0
numpy ≥ 1.21
matplotlib ≥ 3.5
ale_py ≥ 0.8
jupyter ≥ 1.0
```

### Instalación de Dependencias

#### Opción 1: pip local
```bash
# Versión base
pip install gymnasium "gymnasium[atari]" ale_py stable-baselines3 torch

# O con versión CUDA para GPU
pip install gymnasium "gymnasium[atari]" ale_py stable-baselines3 torch --index-url https://download.pytorch.org/whl/cu118
```

#### Opción 2: requirements.txt
```bash
pip install -r requirements.txt
```

---

## 📚 Estructura del Repositorio

```
Reinforced-Learning-amb-Atari-2600/
│
├── README.md                                    # Documentación principal
│
├── Reinforced_Learning_amb_Atari_2600_GuzmanWilliam.ipynb
│   └── Notebook principal con todo el pipeline
│
├── requirements.txt                            # Dependencias
│
├── models/                                     # Modelos entrenados
│   ├── space_invaders_ppo_100k.zip            # Modelo básico
│   ├── space_invaders_ppo_500k.zip            # Modelo optimizado
│   └── training_logs/                         # Registros de entrenamiento
│
├── outputs/                                    # Resultados y gráficos
│   ├── training_rewards.png                   # Gráfico de recompensas
│   ├── episode_lengths.png                    # Duración de episodios
│   └── agent_gameplay/                        # Videos del agente
│
└── notebooks/                                  # Notebooks adicionales (opcional)
    ├── 01_gymnasium_intro.ipynb                # Introducción a Gymnasium
    ├── 02_ppo_algorithm.ipynb                 # Explicación del algoritmo PPO
    └── 03_hyperparameter_tuning.ipynb         # Ajuste de hiperparámetros
```

---

## 🚀 Guía de Inicio Rápido

### Opción 1: Google Colab (Recomendado ⭐)

**Ventajas:**
- ✅ GPU gratis (NVIDIA T4)
- ✅ Sin instalación de dependencias
- ✅ Ejecución rápida
- ✅ Almacenamiento automático

**Pasos:**
1. Abre el notebook en Colab:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/williamG7/Reinforced-Learning-amb-Atari-2600/blob/main/Reinforced_Learning_amb_Atari_2600_GuzmanWilliam.ipynb)

2. Ejecuta la primera celda para instalar dependencias
3. Sigue las celdas en orden

### Opción 2: Ejecución Local

#### Paso 1: Clonar el Repositorio
```bash
git clone https://github.com/williamG7/Reinforced-Learning-amb-Atari-2600.git
cd Reinforced-Learning-amb-Atari-2600
```

#### Paso 2: Crear Entorno Virtual
```bash
# Con venv
python -m venv env
source env/bin/activate  # En Windows: env\Scripts\activate

# O con conda
conda create --name atari-rl python=3.10
conda activate atari-rl
```

#### Paso 3: Instalar Dependencias
```bash
pip install -r requirements.txt

# O instalación manual
pip install gymnasium "gymnasium[atari]" ale_py
pip install stable-baselines3
pip install torch numpy matplotlib jupyter
```

#### Paso 4: Ejecutar Notebook
```bash
jupyter notebook Reinforced_Learning_amb_Atari_2600_GuzmanWilliam.ipynb
```

---

## 📖 Uso y Ejemplos

### 1. Crear Entorno del Juego

```python
import gymnasium as gym
import ale_py

# Registrar entornos de ALE
gym.register_envs(ale_py)

# Crear entorno de Space Invaders
env = gym.make("ALE/SpaceInvaders-v5")

# Resetear y obtener observación inicial
obs, info = env.reset()
print(f"Observación shape: {obs.shape}")  # (210, 160, 3) - imagen RGB
```

### 2. Explorar Acciones

```python
# Número de acciones disponibles
n_actions = env.action_space.n
print(f"Acciones disponibles: {n_actions}")

# Nombres de acciones
action_names = env.unwrapped.get_action_meanings()
for i, action in enumerate(action_names):
    print(f"Acción {i}: {action}")
```

### 3. Ejecutar Juego Manual

```python
obs, info = env.reset()
total_reward = 0

for step in range(500):
    # Tomar una acción aleatoria
    action = env.action_space.sample()
    
    # Ejecutar acción en el entorno
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    # Terminar si el juego acabó
    if terminated or truncated:
        break

print(f"Recompensa total: {total_reward}")
env.close()
```

### 4. Entrenar Agente con PPO

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# Crear entorno con wrappers
env_train = make_atari_env("ALE/SpaceInvaders-v5", n_envs=4, seed=42)

# Frame stacking: apilar 4 frames para captar movimiento
env_train = VecFrameStack(env_train, n_stack=4)

# Crear agente PPO
model = PPO(
    "CnnPolicy",                    # Política convolucional (para imágenes)
    env_train,
    learning_rate=0.0003,           # Tasa de aprendizaje
    n_steps=2048,                   # Pasos antes de actualizar
    batch_size=128,                 # Tamaño de lote
    n_epochs=20,                    # Épocas de entrenamiento
    gamma=0.99,                     # Factor de descuento
    gae_lambda=0.95,                # Ventaja generalizada exponencial
    verbose=1
)

# Entrenar
model.learn(total_timesteps=500000, progress_bar=True)

# Guardar modelo
model.save("space_invaders_ppo")
```

### 5. Evaluar Agente Entrenado

```python
from stable_baselines3 import PPO

# Cargar modelo entrenado
model = PPO.load("space_invaders_ppo")

# Crear entorno de evaluación
eval_env = make_atari_env("ALE/SpaceInvaders-v5", n_envs=1, seed=123)
eval_env = VecFrameStack(eval_env, n_stack=4)

# Jugar varios episodios
episodes = 10
rewards = []

for episode in range(episodes):
    obs, _ = eval_env.reset()
    episode_reward = 0
    done = False
    
    while not done:
        # Usar modelo para predecir acción
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        episode_reward += reward[0]
    
    rewards.append(episode_reward)
    print(f"Episodio {episode+1}: {episode_reward:.0f}")

print(f"Recompensa promedio: {sum(rewards)/len(rewards):.2f}")
```

### 6. Visualizar Entrenamiento

```python
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy

# Cargar datos de entrenamiento
log_dir = "logs/"
x, y = ts2xy(load_results(log_dir), "timesteps")

plt.figure(figsize=(12, 5))

# Gráfico 1: Recompensa vs Timesteps
plt.subplot(1, 2, 1)
plt.plot(x, y, label="Recompensa media")
plt.xlabel("Timesteps")
plt.ylabel("Recompensa")
plt.title("Progreso de Entrenamiento")
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfico 2: Suavizado con media móvil
plt.subplot(1, 2, 2)
window = 100
moving_avg = np.convolve(y, np.ones(window)/window, mode='valid')
plt.plot(x[:len(moving_avg)], moving_avg, label="Media móvil", color='orange')
plt.xlabel("Timesteps")
plt.ylabel("Recompensa (suavizada)")
plt.title("Tendencia de Rendimiento")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_results.png", dpi=150)
plt.show()
```

---

## 📊 Arquitectura de la Red Neuronal

### Política CNN de Stable Baselines3

```
Entrada: Frame de Atari (84x84x4) - 4 frames apilados
         ↓
Convolutional Layer 1: 32 filtros, kernel 8x8, stride 4
         ↓
Convolutional Layer 2: 64 filtros, kernel 4x4, stride 2
         ↓
Convolutional Layer 3: 64 filtros, kernel 3x3, stride 1
         ↓
Flatten: 3136 neuronas
         ↓
Dense Layer 1: 512 neuronas (ReLU)
         ↓
Actor Head: 6 neuronas (una por acción)
Value Head: 1 neurona (estimación del valor del estado)
         ↓
Output: Acción + Valor del Estado
```

### Hiperparámetros Clave

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| **Learning Rate** | 0.0003 | Tasa de aprendizaje del optimizador |
| **Gamma (γ)** | 0.99 | Factor de descuento para recompensas futuras |
| **GAE Lambda** | 0.95 | Parámetro de ventaja generalizada exponencial |
| **N Steps** | 2048 | Pasos antes de actualización de política |
| **Batch Size** | 128 | Tamaño de lote para entrenamiento |
| **N Epochs** | 20 | Épocas de entrenamiento por actualización |
| **Clip Range** | 0.2 | Rango de clipping de PPO |
| **Frame Stack** | 4 | Frames apilados para captar movimiento |

---

## 🎓 Conceptos Clave de Reinforcement Learning

### ¿Qué es RL?

El Aprendizaje por Refuerzo es un paradigma donde un **agente** aprende a tomar **acciones** en un **entorno** para maximizar una **recompensa acumulada**.

```
Entorno              Agente
   ↓                  ↓
[Estado] ←→ [Acción]
   ↓                  ↓
[Recompensa] ← [Política π]
```

### Componentes Principales

1. **Estado (s):** Observación del entorno (imagen del juego)
2. **Acción (a):** Movimiento del agente (0-5)
3. **Recompensa (r):** Feedback del entorno (puntos del juego)
4. **Política (π):** Estrategia que mapea estados a acciones
5. **Valor (V):** Estimación de recompensa futura

### Algoritmo PPO

**Proximal Policy Optimization** es un algoritmo state-of-the-art que:
- Usa **clipping de gradientes** para estabilidad
- Alterna entre recolección de experiencia e actualización de política
- Es más fácil de entrenar que A3C o TRPO

---

## 🔬 Técnicas de Optimización Utilizadas

### 1. Frame Stacking
```python
# Apilar 4 frames consecutivos
# Permite al agente percibir movimiento sin procesamiento temporal
env = VecFrameStack(env, n_stack=4)
```

### 2. Wrappers de Atari
```python
# Aplicados automáticamente por make_atari_env:
# - Preprocesamiento: redimensión a 84x84, escala de grises
# - Episodic life: terminar episodio si pierdes una vida
# - Max pooling: tomar máximo de 2 frames consecutivos
# - Frame skipping: repetir acciones para velocidad
```

### 3. Reward Shaping
```python
# Recompensas personalizadas para guiar al agente
def custom_reward(game_reward, agent_health, enemy_count):
    reward = game_reward
    reward += 0.01 * (100 - enemy_count)  # Bonus por destruir enemigos
    return reward
```

### 4. Normalización y Estandarización
```python
# VecNormalize normaliza observaciones
from stable_baselines3.common.vec_env import VecNormalize
env = VecNormalize(env, norm_obs=True, norm_reward=True)
```

---

## 📈 Resultados y Métricas

### Evolución del Entrenamiento

| Timestep | Recompensa Media | Duración Episodio | Estado |
|----------|-----------------|-------------------|--------|
| 0 | ~152 | 547 pasos | Sin entrenamiento |
| 100K | ~200 | 580 pasos | Primeras mejoras |
| 250K | ~250 | 600+ pasos | Convergencia inicial |
| 500K | ~300+ | 700+ pasos | Excelente desempeño |

### Métricas de Evaluación

```python
# Métricas para evaluar al agente
- Mean Episode Reward: Recompensa promedio por episodio
- Mean Episode Length: Duración promedio de episodios
- Success Rate: % de veces que gana contra enemigos
- Win Rate: % de episodios completados exitosamente
```

---

## 🔧 Troubleshooting

### Error: `gym.error.NamespaceNotFound: Namespace 'ALE' not found`
```python
# Solución: Registrar entornos de ALE
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
```

### Error: `torch.cuda.OutOfMemoryError`
```python
# Reducir tamaño de lote y buffer
model = PPO("CnnPolicy", env, 
            batch_size=32,      # Reducido de 128
            n_steps=1024)       # Reducido de 2048
```

### Entrenamiento muy lento
```bash
# Usar GPU (si disponible)
# En Google Colab: Runtime → Change runtime type → GPU

# O usar múltiples entornos en paralelo
env = make_atari_env("ALE/SpaceInvaders-v5", n_envs=8)  # 8 envs paralelos
```

### El agente no mejora
```python
# Ajustar hiperparámetros
model = PPO("CnnPolicy", env,
    learning_rate=0.0001,       # Más bajo
    n_epochs=4,                 # Menos épocas
    gamma=0.995)                # Factor descuento más alto
```

---

## 📚 Recursos y Referencias

### Teoría de RL
- [OpenAI Spinning Up in Deep RL](https://spinningup.openai.com/) - Excelente introducción
- [DeepMind RL Course](https://www.deepmind.com/learning-resources) - Cursos profesionales
- [Sutton & Barto: RL Book](http://incompleteideas.net/book/the-book-2nd.html) - Referencia clásica

### Librerías y Documentación
- [Stable Baselines3 Docs](https://stable-baselines3.readthedocs.io/) - SB3 completa
- [Gymnasium Documentation](https://gymnasium.farama.org/) - Entornos de juegos
- [ALE Documentation](https://ale.farama.org/) - Atari Learning Environment
- [PyTorch Documentation](https://pytorch.org/docs/) - Deep Learning

### Artículos Clave
- [PPO Paper (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347) - Algoritmo PPO original
- [DQN Paper (Mnih et al., 2015)](https://arxiv.org/abs/1312.5602) - Aprendizaje Q profundo
- [A3C Paper (Mnih et al., 2016)](https://arxiv.org/abs/1602.01783) - Actores-Críticos

### Cursos Online
- [Deep RL Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/home) - UC Berkeley
- [Reinforcement Learning Specialization](https://www.coursera.org/specializations/reinforcement-learning) - Coursera
- [Practical RL Course](https://github.com/yandexdataschool/Practical_RL) - Yandex

---

## 📈 Progreso del Proyecto

- [x] Configuración del entorno Atari 2600
- [x] Exploración de acciones y recompensas
- [x] Implementación de juego manual
- [x] Entrenamiento básico con PPO
- [x] Optimización con frame stacking
- [x] Ajuste de hiperparámetros
- [x] Evaluación del agente entrenado
- [ ] Implementar DQN como comparación
- [ ] Agregar visualización de atención
- [ ] Entrenar en múltiples juegos Atari
- [ ] Crear dashboard de monitoreo en tiempo real
- [ ] Exportar modelo para inferencia en tiempo real

---

## 👥 Contribuciones

¡Contribuciones son bienvenidas! Si quieres mejorar este proyecto:

1. **Fork** el repositorio
2. Crea una rama: `git checkout -b feature/mejora`
3. **Commit** tus cambios: `git commit -m "Add new feature"`
4. **Push:** `git push origin feature/mejora`
5. Abre un **Pull Request**

### Ideas para Contribuir
- Entrenar en otros juegos Atari (Pong, Breakout, etc.)
- Implementar otros algoritmos (DQN, A3C, SAC)
- Crear visualizaciones del proceso de aprendizaje
- Optimizar performance
- Mejorar documentación

**Directrices:**
- Mantén el código documentado
- Sigue PEP 8 para estilo
- Incluye ejemplos de uso
- Actualiza la documentación

---

## 📄 Licencia

Este proyecto está bajo licencia **MIT** y es de libre distribución con fines educativos.

```
MIT License - Reinforced Learning amb Atari 2600

Copyright (c) 2024 William Guzmán

Permiso otorgado, sin costo, a cualquier persona que obtenga una copia
de este software para usarlo con fines educativos y de investigación.
```

**Condiciones de uso:**
- ✅ Úsalo para aprendizaje personal
- ✅ Cita al autor si lo utilizas
- ✅ Comparte mejoras con la comunidad
- ❌ No hagas pasar como tuyo sin mencionar origen
- ❌ Respeta la naturaleza educativa del proyecto

---

## 👨‍💻 Autor

**William Guzmán** | Machine Learning & Game AI Enthusiast

- 🔗 **GitHub:** [@williamG7](https://github.com/williamG7)
- 📧 **Email:** contacto@williamguzman.dev
- 💼 **Intereses:** RL, Game AI, Deep Learning, Computer Vision

---

## ⭐ Apoya el Proyecto

Si te resultó útil:

- ⭐ **Dale una estrella** en GitHub
- 🍴 **Fork** para contribuir o usar como referencia
- 📢 **Comparte** con otros entusiastas de RL
- 💬 **Deja feedback** en las issues
- 🔗 **Enlaza** desde tu proyecto

---

## 📊 Estadísticas del Proyecto

![GitHub stars](https://img.shields.io/github/stars/williamG7/Reinforced-Learning-amb-Atari-2600?style=social)
![GitHub forks](https://img.shields.io/github/forks/williamG7/Reinforced-Learning-amb-Atari-2600?style=social)
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![Stable Baselines3](https://img.shields.io/badge/StableBaselines3-2.0+-green?logo=robot)
![Gymnasium](https://img.shields.io/badge/Gymnasium-Latest-orange?logo=game)
![License MIT](https://img.shields.io/badge/License-MIT-blue)

---

## 🎮 Galería (Ejemplos de Juego)

### Agente Sin Entrenar
- Acciones aleatorias
- Recompensa promedio: ~150 puntos
- No comprende el objetivo

### Agente Después de 100K Timesteps
- Comienza a disparar
- Recompensa promedio: ~200 puntos
- Comportamiento emergente

### Agente Después de 500K Timesteps
- Domina la mecánica del juego
- Recompensa promedio: ~300+ puntos
- Estrategia clara y consistente

---

**Última actualización:** Junio 2024 | **Versión:** 1.0.0

---

### 🔴 NOTAS IMPORTANTES

**Este es un proyecto educativo.**

- Diseñado para aprender cómo funcionan los algoritmos RL
- No es optimizado para producción
- Ideal para experimentación y aprendizaje
- Siéntete libre de modificar, mejorar y adaptar el código

**¿Tienes preguntas?** Abre una issue en GitHub.

---

**¿Te gustó este proyecto? ⭐ Dale una estrella en GitHub y comparte con otros aprendices**
