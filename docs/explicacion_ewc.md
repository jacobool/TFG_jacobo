# Elastic Weight Consolidation (EWC) en el agente de Geometry Dash

## El problema: olvido catastrófico

Cuando una red neuronal aprende una tarea nueva, actualiza sus pesos mediante descenso de gradiente. El problema es que ese proceso no distingue entre pesos que eran importantes para lo que ya sabía y pesos que son irrelevantes. Si entrenas el cubo 2 con el mismo modelo que aprendió el cubo 1, el optimizador sobreescribirá libremente los pesos que hacían funcionar el cubo 1 para adaptarlos al cubo 2. El resultado es que el agente "olvida" el cubo 1 por completo, aunque lo tuviera perfectamente aprendido.

Esto se llama **olvido catastrófico** (*catastrophic forgetting*) y es uno de los problemas fundamentales del aprendizaje continuo (*continual learning*).

---

## Por qué EWC y no las otras alternativas

Antes de llegar a EWC se probaron dos enfoques:

**Congelar el extractor CNN**: se congelaron las capas convolucionales del modelo del cubo 1 y solo se entrenaron las capas finales. El resultado fue malo porque las capas finales (la "cabeza Q") también contienen conocimiento importante sobre cómo actuar en el cubo 1. Al reinicializarlas y entrenarlas solo con datos del cubo 2, el modelo aprendió el cubo 2 pero perdió completamente la capacidad de decidir en el cubo 1.

**Progressive Neural Networks**: se construyó una arquitectura con dos columnas CNN en paralelo: una congelada con los pesos del cubo 1 y otra entrenable para el cubo 2. Aunque la idea es teóricamente sólida, el problema práctico fue el mismo: la cabeza Q (que toma la decisión final sobre qué acción ejecutar) se inicializó aleatoriamente en la nueva arquitectura. Aunque la columna 1 seguía "viendo" bien el cubo 1, el modelo no sabía qué hacer con esa información porque la capa de decisión nunca fue entrenada para el cubo 1.

**EWC** resuelve ambos problemas porque:
- No cambia la arquitectura (es el mismo modelo DQN de siempre)
- No congela ninguna capa completa
- En su lugar, **penaliza los cambios en los pesos que eran importantes para el cubo 1**, permitiendo que los pesos menos importantes se adapten libremente al cubo 2

Con 300k pasos el cubo 1 pasó de 0/10 (progressive) a 6/10 (EWC), lo que confirma que la técnica funciona y simplemente necesita más ajuste.

---

## Cómo funciona EWC

### La intuición

Imagina que los pesos de la red son los "recuerdos" del agente. Algunos recuerdos son críticos para el cubo 1 (si los cambias, el agente deja de saber jugar), y otros son irrelevantes (si los cambias, el agente sigue jugando igual de bien). EWC identifica cuáles son cuáles y pone un "coste" proporcional a la importancia de cada peso cuando el optimizador intenta cambiarlo.

### La Fisher Information Matrix

La herramienta matemática que mide la importancia de cada peso es la **Matriz de Información de Fisher** (FIM). En la práctica no se calcula la matriz completa (sería enormemente costoso), sino su **diagonal**: un número por cada peso que indica cuánto afecta ese peso a las decisiones del agente en el cubo 1.

Matemáticamente, la importancia del peso $\theta_i$ se estima como:

$$F_i = \mathbb{E}\left[\left(\frac{\partial \log \pi(a|s)}{\partial \theta_i}\right)^2\right]$$

Es decir: se hacen $N$ pasos en el entorno del cubo 1, y para cada paso se calcula el gradiente del logaritmo de la política respecto a cada peso. Si ese gradiente es grande en promedio, significa que el peso influye mucho en las decisiones → es importante → merece protección.

En el código:

```python
for _ in range(n_samples):
    q_values  = self.policy.q_net(obs_th)
    log_probs = F.log_softmax(q_values, dim=-1)
    action    = q_values.argmax(dim=-1)
    selected  = log_probs[arange, action].sum()

    selected.backward()  # Calcula gradientes respecto a todos los pesos

    for n, p in self.policy.q_net.named_parameters():
        fisher[n] += p.grad.detach().pow(2)  # Acumula gradiente²

fisher[n] /= n_samples  # Promedia
```

El cuadrado del gradiente es siempre positivo, así que la Fisher acumulada solo crece para los pesos que importan. Al final, `fisher[n]` es un tensor del mismo tamaño que el peso `n` donde cada valor indica la importancia de ese parámetro concreto.

Junto con la Fisher se guardan los **pesos óptimos** $\theta^*$: los valores exactos de los pesos del cubo 1 en el momento del cálculo. Son el "ancla" a la que EWC quiere que los pesos se mantengan cerca.

### La penalización EWC en el loss

Durante el entrenamiento del cubo 2, a cada paso de gradiente se añade un término de penalización al loss de Bellman:

$$\mathcal{L}_{total} = \mathcal{L}_{DQN} + \frac{\lambda}{2} \sum_i F_i \cdot (\theta_i - \theta_i^*)^2$$

Donde:
- $\mathcal{L}_{DQN}$ es el loss estándar de DQN (error de Bellman)
- $F_i$ es la importancia del peso $i$ calculada con la Fisher
- $\theta_i^*$ son los pesos óptimos del cubo 1
- $(\theta_i - \theta_i^*)^2$ es cuánto se ha alejado el peso $i$ de su valor óptimo
- $\lambda$ es el hiperparámetro que controla cuánto importa preservar el cubo 1

En el código:

```python
ewc_penalty = 0
for n, p in self.policy.q_net.named_parameters():
    diff        = p - self.ewc_star_params[n]
    ewc_penalty += (self.ewc_fisher[n] * diff.pow(2)).sum()

loss = dqn_loss + (ewc_lambda / 2) * ewc_penalty
```

El efecto es sutil pero poderoso: si un peso era muy importante para el cubo 1 (Fisher alta), alejarlo de $\theta^*$ genera un coste enorme en el loss, así que el optimizador lo evita. Si un peso era irrelevante para el cubo 1 (Fisher baja), puede cambiar libremente para adaptarse al cubo 2.

---

## Flujo completo del script

### Fase 1: Calcular la Fisher (cubo 1)

El juego debe estar en el nivel del cubo 1. El script:

1. Carga el modelo entrenado del cubo 1
2. Ejecuta 500–800 pasos observando el cubo 1
3. En cada paso calcula el gradiente del log de la política y lo eleva al cuadrado
4. Promedia los gradientes² → **Fisher diagonal**
5. Guarda los pesos actuales como $\theta^*$
6. Persiste la Fisher en disco (`cubo2_ewc_fisher.pt`) para no tener que recalcularla

### Fase 2: Entrenar en el cubo 2 con EWC activo

El usuario reposiciona el juego al nivel personalizado del cubo 2. A partir de aquí el entrenamiento es DQN normal excepto por el loss modificado. El agente:

- Juega el cubo 2, acumula experiencias en el replay buffer
- Cada 4 pasos hace un paso de gradiente con el loss EWC
- Los pesos importantes para el cubo 1 se mantienen cerca de $\theta^*$
- Los pesos menos importantes se adaptan al cubo 2

---

## Hiperparámetros y su efecto

### `EWC_LAMBDA` — cuánto importa preservar el cubo 1

Es el parámetro más crítico. Controla el equilibrio entre aprender el cubo 2 y no olvidar el cubo 1:

| Lambda | Efecto |
|--------|--------|
| Demasiado bajo (< 2000) | El cubo 1 se olvida (la penalización no compensa los gradientes del cubo 2) |
| Óptimo (5000–10000) | El cubo 1 se preserva y el cubo 2 se aprende gradualmente |
| Demasiado alto (> 20000) | El cubo 2 no se aprende (los pesos están demasiado anclados) |

En el script inicial se usó 5000 y se consiguió 6/10 en cubo 1. En la fase 2 se sube a 8000 para mejorar la consistencia.

### `EWC_FISHER_SAMPLES` — precisión de la Fisher

Más muestras = Fisher más precisa = mejor identificación de qué pesos son importantes. Con 500 muestras hay ruido estadístico. Con 800 la estimación es más estable. Rara vez merece la pena ir más allá de 1000 porque el coste computacional sube y la mejora marginal es pequeña.

### `learning_rate` — velocidad de cambio de los pesos

Con EWC se usa un LR más bajo que en el entrenamiento inicial (2e-5 → 8e-6) porque:
- Los pesos ya tienen una buena base del cubo 1
- Cambios pequeños permiten que la penalización EWC sea efectiva
- Con LR alto, el gradiente del cubo 2 puede superar la penalización en un solo paso

### `exploration_final_eps` — exploración residual

Se baja de 0.02 a 0.005 porque:
- En cubo 1, el agente ya sabe qué hacer: no necesita explorar
- Una acción aleatoria en el momento equivocado del cubo 1 cuenta como fallo aunque los pesos sean correctos
- En cubo 2, con 0.005 hay suficiente exploración para aprender patrones nuevos sin desestabilizar lo aprendido

---

## Por qué la Fisher se guarda en disco

Calcular la Fisher requiere que el juego esté en el nivel del cubo 1. En sesiones de entrenamiento posteriores (fase 2, seguirentreno), el juego debe estar en el cubo 2. Si la Fisher se recalculara cada vez, habría que cambiar de nivel manualmente al inicio de cada sesión.

Al persistirla en `cubo2_ewc_fisher.pt`, el script la carga directamente en sesiones posteriores. Esto también garantiza que siempre se usa la Fisher del modelo correcto (el cubo 1 original), no una versión contaminada por el entrenamiento del cubo 2.

---

## Señales para monitorizar el entrenamiento

### `ewc_penalty` en los logs

La penalización EWC aparece en los logs como `train/ewc_penalty`. Su comportamiento esperado:

- **Al inicio**: sube porque el modelo empieza a aprender el cubo 2 y los pesos se alejan de $\theta^*$
- **A medida que converge**: se estabiliza en un valor constante
- **Si sube indefinidamente**: el lambda es demasiado bajo y el cubo 1 se está olvidando
- **Si es casi cero siempre**: el lambda es demasiado alto y el cubo 2 no está aprendiendo nada

### Consistencia en cubo 1

La métrica real de éxito es cuántas veces de 10 el agente pasa el cubo 1 al probar en el nivel completo:

| Resultado | Diagnóstico |
|-----------|-------------|
| 0–3/10 | Lambda demasiado bajo, subir |
| 4–7/10 | En buen camino, seguir entrenando |
| 8–10/10 | EWC funcionando correctamente |

---

## Limitaciones de EWC

EWC no es magia: tiene limitaciones importantes que conviene conocer.

**La Fisher es una aproximación diagonal.** La verdadera FIM es una matriz completa que captura interacciones entre pesos. La diagonal ignora esas interacciones. En redes grandes como NatureCNN esto introduce error, pero en la práctica la aproximación diagonal es suficiente para la mayoría de casos.

**Los pesos óptimos son un punto, no una región.** EWC ancla los pesos a los valores exactos que tenían al calcular la Fisher. Si el entrenamiento previo del cubo 1 no había convergido completamente, $\theta^*$ puede no ser el óptimo real, y la Fisher puede estar sobreestimando la importancia de pesos que en realidad eran accidentales.

**EWC asume que las tareas son independientes.** Si el cubo 1 y el cubo 2 requieren representaciones internas contradictorias (por ejemplo, el cubo 2 tiene patrones visuales que el cubo 1 nunca vio y que son confusos para los pesos preservados), la penalización puede impedir aprender el cubo 2 aunque el lambda sea bajo.

**No escala bien con muchas tareas.** Para dos tareas (cubo 1 y cubo 2) EWC funciona bien. Si hubiera cinco o diez tareas distintas, la suma de penalizaciones se volvería inmanejable. En ese caso se usarían variantes como Online EWC o Progress & Compress.

---

## Comparativa de enfoques probados

| Técnica | Cubo 1 preservado | Cubo 2 aprendido | Complejidad |
|---------|-------------------|------------------|-------------|
| Congelar CNN | No (cabeza Q olvidada) | Parcialmente | Baja |
| Progressive Networks | No (cabeza Q aleatoria) | Sí | Alta |
| **EWC** | **Sí (6/10 → objetivo 8-9/10)** | **Sí** | **Media** |
| Entorno unificado completo | Sí (natural) | Sí (natural) | Baja |

El entorno unificado (`gd_rl_completo.py`) sigue siendo la opción más robusta a largo plazo porque el replay buffer mezcla naturalmente experiencias de todas las secciones. EWC es la mejor opción si se quiere entrenar las secciones por separado y mantener el control preciso sobre qué se preserva y qué se modifica.
