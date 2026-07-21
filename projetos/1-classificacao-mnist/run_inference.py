import os
import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# Projeto 1 — Inferência com o Modelo Otimizado (model.tflite)
# ---------------------------------------------------------------------------

N_SAMPLES = 5

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model.tflite")
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = np.expand_dims(x_test.astype("float32") / 255.0, axis=-1)

    print(f"Executando inferência em {N_SAMPLES} amostras:\n")
    for i in range(N_SAMPLES):
        sample = np.expand_dims(x_test[i], axis=0).astype(input_details[0]["dtype"])
        
        interpreter.set_tensor(input_details[0]["index"], sample)
        interpreter.invoke()
        
        output = interpreter.get_tensor(output_details[0]["index"])[0]
        pred_class = int(np.argmax(output))
        
        print(f"Amostra {i + 1}: predito = {pred_class} | real = {y_test[i]}")

if __name__ == "__main__":
    main()