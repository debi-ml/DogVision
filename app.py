import gradio as gr
import tensorflow as tf
from huggingface_hub import from_pretrained_keras
from tensorflow.keras import mixed_precision

# Load your trained models
model = from_pretrained_keras("ml-debi/EfficientNetV2S-StanfordDogsA")

# Add information about the models
model_info = """
### Model Information

"""

examples = [["./examples/nala.jpg"], ["./examples/molly.jpg"]]

def preprocess(image):
    print("before resize", image.shape)
    image = tf.image.resize(image, [224, 224])
    
    image = tf.expand_dims(image, axis=0)
    print("After expanddims", image.shape)
    return image

def predict(image):

    if mixed_precision.global_policy() == "mixed_float16":
        mixed_precision.set_global_policy(policy="float32")

    image = preprocess(image)
    print(mixed_precision.global_policy())
    prediction = model.predict(image)[0]
    print("model prediction", prediction)
    confidences = {model.config['id2label'][str(i)]: float(prediction[i]) for i in range(101)}
    return confidences

iface = gr.Interface(
    fn=predict,
    inputs=[gr.Image()],
    outputs=[gr.Label(num_top_classes=5)],
    title="Dog Vision Mini Project",
    description=f"{model_info}\n",
    examples=examples
)

iface.launch()