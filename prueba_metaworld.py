import metaworld
import random
import time

print("--- INICIANDO ENTORNO ---")

# 1. Cargamos una tarea sencilla (alcanzar un punto)
ml1 = metaworld.ML1('reach-v3') 
env = ml1.train_classes['reach-v3'](render_mode='human')
task = random.choice(ml1.train_tasks)
env.set_task(task)

obs, _ = env.reset()

print("--- ¡ENTRANDO EN EL BUCLE! Deberías ver datos abajo ---")
time.sleep(1) # Pausa de 1 segundo para que leas la consola

for step in range(3000):
    # Generamos una acción aleatoria
    action = env.action_space.sample() 
    
    # Ejecutamos el paso
    obs, reward, terminated, truncated, info = env.step(action)
    
    # ESTAS LÍNEAS SON LAS QUE ESCRIBEN EN TU TERMINAL:
    print(f"PASO: {step:3} | RECOMPENSA: {reward:8.5f} | ÉXITO: {info['success']}")
    
    # Dibujamos en la ventana
    env.render()

print("--- FIN DE LA PRUEBA ---")
env.close()